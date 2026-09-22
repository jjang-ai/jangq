"""Bind resumable GPTQ outputs to their actual calibration inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recipe_sha256(plan):
    # Labels/allocation reports/media flags do not change the quantization.
    keys = ("schema", "source_revision", "expert_default", "experts", "awq_alpha", "stats")
    recipe = {k: plan[k] for k in keys if k in plan}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def bind_run(out_dir, src, tokens, plan, *, tau_scale, damp):
    out_dir = Path(out_dir)
    expected = {
        "schema": "mimo-v26-gptq-run-v1",
        "recipe_sha256": recipe_sha256(plan),
        "stats_sha256": file_sha256(plan["stats"]),
        "tokens_sha256": hashlib.sha256(tokens.tobytes(order="C")).hexdigest(),
        "tokens_shape": list(tokens.shape), "tokens_dtype": str(tokens.dtype),
        "source_config_sha256": file_sha256(Path(src) / "config.json"),
        "source_index_sha256": file_sha256(Path(src) / "model.safetensors.index.json"),
        "tau_scale": tau_scale, "damp": damp,
    }
    path = out_dir / "gptq_run.json"
    if path.exists():
        actual = json.loads(path.read_text())
        changed = [k for k, v in expected.items() if actual.get(k) != v]
        if changed:
            raise ValueError(f"GPTQ resume inputs changed: {changed}; use a separate output directory")
    else:
        if any(out_dir.glob("L*.safetensors")):
            raise ValueError("GPTQ outputs lack run provenance; use a separate output directory")
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2) + "\n")
    return expected


def validate_conversion(gptq_dir, plan, src):
    path = Path(gptq_dir) / "gptq_run.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "mimo-v26-gptq-run-v1":
        raise ValueError("Unknown GPTQ run provenance")
    if manifest.get("recipe_sha256") != recipe_sha256(plan):
        raise ValueError("GPTQ codes were solved for a different quantization plan")
    if manifest.get("stats_sha256") != file_sha256(plan["stats"]):
        raise ValueError("GPTQ codes were solved with different calibration statistics")
    for key, name in (("source_config_sha256", "config.json"),
                      ("source_index_sha256", "model.safetensors.index.json")):
        if manifest.get(key) != file_sha256(Path(src) / name):
            raise ValueError(f"GPTQ source metadata changed: {name}")
    sequential_path = Path(gptq_dir) / "sequential_run.json"
    if sequential_path.exists():
        sequential = json.loads(sequential_path.read_text())
        if sequential.get("schema") != "mimo-v26-sequential-gptq-v1" or sequential.get("max_layers") != 0:
            raise ValueError("Unknown or incomplete sequential GPTQ run")
        if sequential.get("tokens_sha256") != manifest.get("tokens_sha256"):
            raise ValueError("Sequential GPTQ token provenance disagrees")
        expected_files = {"v26_sequential_gptq.py", "v26_gptq.py", "v26_quant.py",
                          "v26_model.py", "v26_source.py", "v26_sweep.py"}
        hashes = sequential.get("source_files_sha256", {})
        if set(hashes) != expected_files:
            raise ValueError("Incomplete sequential source provenance")
        for name, digest in hashes.items():
            if digest != file_sha256(Path(__file__).with_name(name)):
                raise ValueError(f"Sequential GPTQ implementation changed: {name}")
        cfg = json.loads((Path(src) / "config.json").read_text())
        capture = json.loads((Path(gptq_dir) / "hessian_capture_report.json").read_text())
        if set(capture) != {str(i) for i in range(cfg["num_hidden_layers"])} or not all(
                row.get("completed") for row in capture.values()):
            raise ValueError("Sequential GPTQ layer capture is incomplete")
        expected = {}
        for layer, moe in enumerate(cfg["moe_layer_freq"]):
            if not moe:
                continue
            for projection in ("gate_proj", "up_proj", "down_proj"):
                spec = plan.get("experts", {}).get(str(layer), {}).get(projection, plan["expert_default"])
                if spec["mode"] == "affine":
                    expected[f"L{layer}.{projection}"] = spec
        report = json.loads((Path(gptq_dir) / "gptq_report.json").read_text())
        if set(report) != set(expected):
            raise ValueError("Sequential GPTQ projection census mismatch")
        for name, spec in expected.items():
            row = report[name]
            if row.get("spec") != spec or row.get("experts") != cfg["n_routed_experts"]:
                raise ValueError(f"Sequential GPTQ projection recipe mismatch: {name}")
            if row.get("sha256") != file_sha256(Path(gptq_dir) / (name + ".safetensors")):
                raise ValueError(f"Sequential GPTQ projection checksum mismatch: {name}")
    return manifest


def describe_run(gptq_dir):
    """Public-safe method metadata; call validate_conversion before converting."""
    result = {
        "applied": bool(gptq_dir),
        "hessian": "per-expert E[x x^T] of routed tokens, shrunk to the layer pool (tau = d tokens), 1% damping",
        "grid": "fixed = imatrix fit (bytes identical to non-GPTQ build)",
        "guard": "per-expert keep GPTQ only if Hessian-weighted error beats the fitted RTN codes",
        "propagation": "source activations between layers",
        "report": "gptq_report.json alongside the codes",
    }
    if not gptq_dir:
        return result
    directory = Path(gptq_dir)
    sequential_path = directory / "sequential_run.json"
    if sequential_path.exists():
        sequential = json.loads(sequential_path.read_text())
        if sequential.get("schema") != "mimo-v26-sequential-gptq-v1":
            raise ValueError("Unknown sequential GPTQ schema")
        result.update(
            propagation=sequential["propagation"], grid=sequential["grid"],
            guard=sequential["selection"],
            incumbent_manifest_sha256=sequential["incumbent_manifest_sha256"],
            sequential_run_sha256=file_sha256(sequential_path),
            report="quantization/gptq_report.json",
        )
    return result
