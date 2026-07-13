"""Verify inspect-source returns valid JSON with expected keys."""
import json
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_qwen"


def test_inspect_source_prints_valid_json():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(FIXTURE)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["model_type"] == "qwen3_5_moe"
    assert "text_model_type" in data
    assert data["is_moe"] is True
    assert data["num_experts"] == 8
    assert "num_experts_per_tok" in data
    assert data["dtype"] in ("bfloat16", "float16", "float8_e4m3fn", "unknown")
    assert "jangtq_compatible" in data
    assert data["jangtq_compatible"] is True   # qwen3_5_moe is in the v1 whitelist
    assert "is_video_vl" in data
    assert data["is_video_vl"] is False   # tiny_qwen fixture has no video_preprocessor_config.json
    assert "has_generation_config" in data


def test_inspect_source_video_vl_false_for_non_video_fixture():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(FIXTURE)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["is_video_vl"] is False
    assert "num_hidden_layers" in data
    assert data["num_hidden_layers"] == 2   # tiny_qwen fixture value


def test_inspect_source_reads_nested_qwen36_text_config(tmp_path):
    model_dir = tmp_path / "qwen36"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
        },
    }))

    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(model_dir)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["model_type"] == "qwen3_5_moe"
    assert data["text_model_type"] == "qwen3_5_moe_text"
    assert data["is_moe"] is True
    assert data["num_experts"] == 256
    assert data["num_hidden_layers"] == 40
    assert data["num_experts_per_tok"] == 8
    assert data["hidden_size"] == 2048
    assert data["jangtq_compatible"] is True


def test_inspect_source_missing_config_errors(tmp_path):
    (tmp_path / "README").write_text("nope")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode != 0
    assert "config.json" in r.stderr.lower()


def _assert_clean_error(r: subprocess.CompletedProcess, *, expect_phrase: str) -> None:
    """Shared invariants for inspect-source failure surfaces (M120).

    A well-formed error must:
      1. exit non-zero
      2. NOT print a Python traceback (no `Traceback (most recent call last)`)
      3. include the phrase describing WHAT went wrong so the wizard can
         relay it to the user in plain English
    """
    assert r.returncode != 0
    assert "Traceback" not in r.stderr, (
        "inspect-source leaked a Python traceback — "
        "user would see a cryptic multi-line stacktrace:\n" + r.stderr
    )
    assert expect_phrase in r.stderr.lower(), (
        f"stderr missing phrase {expect_phrase!r}, got:\n{r.stderr}"
    )


def test_inspect_source_malformed_json_errors_cleanly(tmp_path):
    """M120: a corrupt config.json must not crash with a bare JSONDecodeError."""
    (tmp_path / "config.json").write_text("{ this is not json")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    _assert_clean_error(r, expect_phrase="config.json")


def test_inspect_source_empty_config_errors_cleanly(tmp_path):
    """M120: empty config.json (zero-byte or whitespace only) is a common disk-
    failure mode — don't surface `Expecting value: line 1 column 1 (char 0)`."""
    (tmp_path / "config.json").write_text("")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    _assert_clean_error(r, expect_phrase="config.json")


def test_inspect_source_non_dict_config_errors_cleanly(tmp_path):
    """M120: valid JSON but not an object (e.g. someone dumps a list) used to
    AttributeError on `cfg.get(...)` — must fail with a clean diagnostic."""
    (tmp_path / "config.json").write_text("[1, 2, 3]")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    _assert_clean_error(r, expect_phrase="config.json")


# --- M166 (iter 89): HF hub cache layout — symlinked shards ---
#
# huggingface_hub's `snapshot_download` creates a cache layout where the
# user-visible snapshot directory (`~/.cache/huggingface/hub/models--org--
# name/snapshots/<hash>/`) contains SYMLINKS to real blobs under
# `../../blobs/`. JANG Studio users who point SourceStep at a snapshot
# directory rely on this working transparently. The inspect_source
# implementation currently WORKS for symlinks (pathlib.glob matches them,
# Path.stat() follows them, open() follows them) but there's no test
# verifying this behavior. A future perf-motivated refactor (e.g.
# swapping to lstat, or filtering out symlinks for security reasons)
# could silently break HF-hub users.
#
# These tests lock in the HF-cache-layout contract.


def _make_safetensors_shard(path: Path, n_bytes: int) -> None:
    """Write a minimal valid safetensors file (8-byte header + empty JSON)."""
    import struct
    header = b'{"__metadata__": {}}'
    payload = b"\x00" * max(0, n_bytes - 8 - len(header))
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        fh.write(payload)


def test_inspect_source_handles_symlinked_shards(tmp_path):
    """HF snapshot dirs symlink .safetensors to blobs. stat() + glob both
    follow symlinks, but pin that behavior in a regression test."""
    # Real blobs live in blobs/
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    real_shard = blobs / "abc123_real_blob"
    _make_safetensors_shard(real_shard, n_bytes=4096)

    # Snapshot dir mimics the HF hub layout: config.json + symlinked shard.
    snap = tmp_path / "snapshot_hash_deadbeef"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_experts": 8,
    }))
    (snap / "model-00001-of-00001.safetensors").symlink_to(real_shard)

    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(snap)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["shard_count"] == 1, (
        "symlinked shard should be counted by glob — HF cache layout regression"
    )
    assert data["total_bytes"] == 4096, (
        f"total_bytes should be the TARGET size (4096), got {data['total_bytes']} — "
        "lstat-style regression; HF-hub users would see 0-byte models"
    )
    assert data["model_type"] == "qwen3_5_moe"


def test_inspect_source_broken_symlink_emits_clean_error(tmp_path):
    """M167 (iter 90): HF cache with a git-gc'd blob leaves a dangling
    symlink in the snapshot dir. Pre-M167, inspect_source raised a bare
    FileNotFoundError traceback when stat() followed the dangling link —
    user saw cryptic multi-line Python stack with no hint of what to do.
    Post-M167, emit a plain-English "this shard is a broken symlink,
    re-download the model" message and exit non-zero. Matches iter-21 M120
    "no cryptic tracebacks" contract."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "num_hidden_layers": 2,
        "hidden_size": 128,
    }))
    # Dangling symlink: target never existed.
    broken = tmp_path / "model-00001-of-00001.safetensors"
    broken.symlink_to(tmp_path / "nonexistent_blob")

    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    _assert_clean_error(r, expect_phrase="broken symlink")
    # Also surface the specific shard name so the user can identify what's broken.
    assert "model-00001-of-00001.safetensors" in r.stderr, (
        f"stderr must name the broken shard so user can locate it in the cache. Got:\n{r.stderr}"
    )


def test_inspect_source_handles_symlinked_directory(tmp_path):
    """User points at a symlinked directory (e.g. ~/my-model -> ~/.cache/hf/…/snapshot).
    Both the directory AND the shards-in-directory may be symlinks."""
    real_model = tmp_path / "real_model"
    real_model.mkdir()
    (real_model / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "num_hidden_layers": 2,
        "hidden_size": 128,
    }))
    _make_safetensors_shard(real_model / "model.safetensors", n_bytes=2048)

    # Symlink the whole directory.
    sym_dir = tmp_path / "sym_model"
    sym_dir.symlink_to(real_model, target_is_directory=True)

    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(sym_dir)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["shard_count"] == 1
    assert data["total_bytes"] == 2048
    assert data["model_type"] == "qwen3_5_moe"
