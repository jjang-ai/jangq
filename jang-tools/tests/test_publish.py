"""Tests for jang_tools.publish.

We never actually hit HF here — use --dry-run to exercise the code path.
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.fixture
def fake_converted(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "_name_or_path": "Qwen/Qwen3-0.6B"}))
    (d / "jang_config.json").write_text(json.dumps({
        "format": "jang", "family": "jang", "profile": "JANG_4K",
        "quantization": {"actual_bits_per_weight": 4.23, "block_size": 64, "bit_widths_used": [4]},
    }))
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x"}))
    (d / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1000)
    return d


def test_cli_rejects_missing_token(fake_converted):
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "publish",
         "--model", str(fake_converted), "--repo", "test/model"],
        capture_output=True, text=True, check=False, env=env,
    )
    assert r.returncode == 2
    assert "HF_HUB_TOKEN" in r.stderr


def test_cli_dry_run(fake_converted, monkeypatch):
    monkeypatch.setenv("HF_HUB_TOKEN", "dummy_token")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "publish",
         "--model", str(fake_converted), "--repo", "test/model", "--dry-run", "--json"],
        capture_output=True, text=True, check=False,
        env={**os.environ, "HF_HUB_TOKEN": "dummy_token"},
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["dry_run"] is True
    assert data["repo"] == "test/model"
    assert data["files_count"] >= 4   # config, jang_config, tokenizer_config, safetensors, README (generated)


def test_cli_rejects_literal_token_via_argv(fake_converted):
    """Security regression test for M41: literal tokens on argv leak via `ps aux`.
    The CLI should reject a --token VALUE that isn't a file path and instead
    direct users to set HF_HUB_TOKEN.
    """
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "publish",
         "--model", str(fake_converted), "--repo", "test/model",
         "--token", "hf_literal_looking_token_abc123xyz",  # not a file path
         "--dry-run"],
        capture_output=True, text=True, check=False, env=env,
    )
    assert r.returncode == 2
    assert "must be a FILE PATH" in r.stderr or "HF_HUB_TOKEN" in r.stderr
    # Importantly: the literal token value must NOT be echoed back in stderr
    # (which would defeat the purpose — it would appear in shell history + logs).
    assert "hf_literal_looking_token_abc123xyz" not in r.stderr


def test_publish_rejects_familyless_authoritative_stamp_before_token_or_card(
    fake_converted,
):
    (fake_converted / "jang_config.json").write_text(json.dumps({
        "format": "JANG",
        "capabilities": {
            "reasoning_parser": "qwen3",
            "tool_parser": "qwen",
            "think_in_template": True,
            "supports_tools": True,
            "supports_thinking": True,
            "modality": "text",
            "cache_type": "kv",
        },
    }))
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jang_tools",
            "publish",
            "--model",
            str(fake_converted),
            "--repo",
            "test/model",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 2
    assert "capabilities.family" in result.stderr
    assert "HF_HUB_TOKEN" not in result.stderr
    assert not (fake_converted / "README.md").exists()


def test_cli_accepts_token_file(fake_converted, tmp_path):
    """--token FILEPATH should read the file and proceed (dry-run).

    This verifies the file-path branch still works after the literal-token
    rejection was added.
    """
    token_file = tmp_path / "token.txt"
    token_file.write_text("hf_dummy_token_for_test\n")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "publish",
         "--model", str(fake_converted), "--repo", "test/model",
         "--token", str(token_file),
         "--dry-run", "--json"],
        capture_output=True, text=True, check=False, env=env,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["dry_run"] is True


# ────────────────────────────────────────────────────────────────────
# Iter 23: M43 — per-file upload emits JSONL progress
# ────────────────────────────────────────────────────────────────────

class _FakeUploadFile:
    """Capture upload_file calls so tests can assert on the interaction
    shape without hitting the network."""
    def __init__(self):
        self.calls = []

    def __call__(self, *, path_or_fileobj, path_in_repo, repo_id, token, commit_message):
        self.calls.append({
            "path": path_or_fileobj,
            "repo_path": path_in_repo,
            "repo": repo_id,
            "token_redacted": bool(token),
            "commit_message": commit_message,
        })


def test_upload_with_progress_iterates_every_file(tmp_path):
    from jang_tools.publish import _upload_with_progress
    from jang_tools.progress import ProgressEmitter

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"x" * 100)
    (model_dir / "tokenizer.json").write_text("{}")

    fake_upload = _FakeUploadFile()
    # Don't emit to real stderr during the test — capture via StringIO.
    err = io.StringIO()
    emitter = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=err)

    url = _upload_with_progress(
        model_dir=model_dir,
        repo_id="test/model",
        token="tok",
        emitter=emitter,
        commit_message="upload test",
        upload_file=fake_upload,
    )
    assert url == "https://huggingface.co/test/model"
    # Every file must have been uploaded exactly once.
    uploaded_paths = sorted(c["repo_path"] for c in fake_upload.calls)
    assert uploaded_paths == ["config.json", "model-00001-of-00001.safetensors", "tokenizer.json"]
    # Commit message carries the idx/total progress string per file.
    assert all("(1/3:" in c["commit_message"] or
               "(2/3:" in c["commit_message"] or
               "(3/3:" in c["commit_message"] for c in fake_upload.calls)


def test_upload_with_progress_emits_expected_jsonl_shape(tmp_path):
    """Verify the stderr JSONL stream matches the convert 5-phase protocol
    enough for Swift's JSONLProgressParser to consume it. Specifically:
    3 phase events + info + ≥1 tick events, all valid JSON lines."""
    from jang_tools.publish import _upload_with_progress
    from jang_tools.progress import ProgressEmitter

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for i in range(5):
        (model_dir / f"file-{i}.bin").write_bytes(b"y" * 1000)

    err = io.StringIO()
    emitter = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=err)
    _upload_with_progress(
        model_dir=model_dir, repo_id="t/m", token="tok",
        emitter=emitter, commit_message="msg",
        upload_file=_FakeUploadFile(),
    )
    lines = [line for line in err.getvalue().splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    types = [e["type"] for e in events]
    assert types.count("phase") == 3, f"expected exactly 3 phase events, got {types}"
    assert "info" in types, "must emit an info event with file count + size"
    assert types.count("tick") >= 1, "must emit at least one tick (the final 100% one)"
    # All events must have a `v` schema version and a `ts` timestamp
    for e in events:
        assert e.get("v") == 1, f"every event must carry v=1, got {e}"
        assert isinstance(e.get("ts"), (int, float)), f"every event needs ts, got {e}"


def test_upload_with_progress_raises_on_empty_dir(tmp_path):
    """Empty model dir = nothing to upload; must raise rather than silently
    create a repo with no files."""
    from jang_tools.publish import _upload_with_progress
    from jang_tools.progress import ProgressEmitter
    model_dir = tmp_path / "empty"
    model_dir.mkdir()
    err = io.StringIO()
    emitter = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=err)
    with pytest.raises(RuntimeError, match="no files to upload"):
        _upload_with_progress(
            model_dir=model_dir, repo_id="t/m", token="tok",
            emitter=emitter, commit_message="msg",
            upload_file=_FakeUploadFile(),
        )


def test_upload_excludes_jang_imatrix(tmp_path):
    """M114 (iter 38): jang_imatrix.safetensors must NOT be uploaded to HF.
    It's useful locally as a convert cache but is pure bloat on a published
    model — per feedback_model_checklist.md's "155 GB bloat" incident.
    """
    from jang_tools.publish import _upload_with_progress
    from jang_tools.progress import ProgressEmitter
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1000)
    # The junk file that must NOT upload
    (model_dir / "jang_imatrix.safetensors").write_bytes(b"y" * 9999)

    fake_upload = _FakeUploadFile()
    err = io.StringIO()
    emitter = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=err)
    _upload_with_progress(
        model_dir=model_dir, repo_id="test/m", token="tok",
        emitter=emitter, commit_message="msg",
        upload_file=fake_upload,
    )
    uploaded_names = {c["repo_path"] for c in fake_upload.calls}
    assert "jang_imatrix.safetensors" not in uploaded_names, \
        "imatrix is local-cache-only; must not be uploaded to HF"
    assert "config.json" in uploaded_names
    assert "model-00001-of-00001.safetensors" in uploaded_names


def test_dry_run_excludes_jang_imatrix_from_size(fake_converted, monkeypatch):
    """Dry-run preview must match what actually uploads. Pre-M114, user saw
    N files / X GB in preview but N-1 files / X-imatrix_size GB actually
    uploaded — confusing."""
    # Add an imatrix to the fixture
    (fake_converted / "jang_imatrix.safetensors").write_bytes(b"z" * 50_000)
    monkeypatch.setenv("HF_HUB_TOKEN", "dummy")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "publish",
         "--model", str(fake_converted), "--repo", "test/model",
         "--dry-run", "--json"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "HF_HUB_TOKEN": "dummy"},
    )
    data = json.loads(r.stdout)
    # The 50000-byte imatrix must NOT be counted in total_size_bytes
    # (fixture has ~1000 bytes of real content, so total < 50_000)
    assert data["total_size_bytes"] < 50_000, \
        f"imatrix leaked into dry-run size: {data['total_size_bytes']}"


def test_publish_cli_has_progress_flag():
    """The --progress=json flag is the Swift-side contract. Pin it in the
    help output so a rename would break the Swift PublishService integration
    noisily (once that lands — M43's Swift portion is iter-24 work)."""
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "publish", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--progress" in r.stdout
    assert "json" in r.stdout


def test_dry_run_generates_readme(fake_converted, monkeypatch):
    monkeypatch.setenv("HF_HUB_TOKEN", "dummy")
    assert not (fake_converted / "README.md").exists()
    subprocess.run(
        [sys.executable, "-m", "jang_tools", "publish",
         "--model", str(fake_converted), "--repo", "test/model", "--dry-run"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "HF_HUB_TOKEN": "dummy"},
    )
    assert (fake_converted / "README.md").exists()
    card = (fake_converted / "README.md").read_text()
    assert "license:" in card
    assert "base_model:" in card
