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
    return manifest
