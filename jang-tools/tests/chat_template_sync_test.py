"""Pins for the bundle chat-template consistency helper.

Motivated by four real divergent bundles found on the model drive
2026-08-16 (Zaya-8B-JANG_4M and three Nemotron-Omni-Nano variants).
"""

import json

from jang_tools.chat_template_sync import (
    TEMPLATE_CONSISTENT,
    TEMPLATE_DIVERGENT,
    TEMPLATE_EMBEDDED_ONLY,
    TEMPLATE_JINJA_ONLY,
    TEMPLATE_NONE,
    audit_bundle_chat_template,
    sync_bundle_chat_template,
)

TPL = "{% for m in messages %}{{ m.content }}{% endfor %}"
FIXED = TPL + "{# items fix #}"


def _bundle(tmp_path, *, jinja=None, embedded=None, extra=None):
    if jinja is not None:
        (tmp_path / "chat_template.jinja").write_text(jinja)
    config = dict(extra or {})
    if embedded is not None:
        config["chat_template"] = embedded
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    return tmp_path


def test_audit_flags_the_zaya_shape(tmp_path):
    """The .jinja carries the fix; the embedded copy is the pre-fix text."""
    b = _bundle(tmp_path, jinja=FIXED, embedded=TPL)
    audit = audit_bundle_chat_template(b)
    assert audit["status"] == TEMPLATE_DIVERGENT
    assert audit["jinja"] != audit["embedded"]


def _sub(tmp_path, name, **kw):
    d = tmp_path / name
    d.mkdir()
    return _bundle(d, **kw)


def test_audit_classifies_the_other_shapes(tmp_path):
    cases = {
        "consistent": (_sub(tmp_path, "a", jinja=TPL, embedded=TPL), TEMPLATE_CONSISTENT),
        "jinja_only": (_sub(tmp_path, "b", jinja=TPL), TEMPLATE_JINJA_ONLY),
        "embedded_only": (_sub(tmp_path, "c", embedded=TPL), TEMPLATE_EMBEDDED_ONLY),
        "none": (_sub(tmp_path, "d"), TEMPLATE_NONE),
    }
    for label, (bundle, expected) in cases.items():
        assert audit_bundle_chat_template(bundle)["status"] == expected, label


def test_whitespace_only_difference_is_not_divergence(tmp_path):
    b = _bundle(tmp_path, jinja=TPL + "\n\n", embedded="  " + TPL)
    assert audit_bundle_chat_template(b)["status"] == TEMPLATE_CONSISTENT


def test_sync_rewrites_the_embedded_copy_from_the_jinja(tmp_path):
    b = _bundle(tmp_path, jinja=FIXED, embedded=TPL, extra={"model_max_length": 4096})
    result = sync_bundle_chat_template(b)

    assert result["changed"] is True
    assert result["status_after"] == TEMPLATE_CONSISTENT

    config = json.loads((b / "tokenizer_config.json").read_text())
    assert config["chat_template"] == FIXED
    # unrelated keys must survive the rewrite
    assert config["model_max_length"] == 4096
    # the .jinja is authoritative and must NOT be touched
    assert (b / "chat_template.jinja").read_text() == FIXED


def test_sync_backs_up_the_original_once(tmp_path):
    b = _bundle(tmp_path, jinja=FIXED, embedded=TPL)
    first = sync_bundle_chat_template(b)
    backup = b / "tokenizer_config.json.bak-pre-template-sync"
    assert backup.exists()
    assert json.loads(backup.read_text())["chat_template"] == TPL

    # a second run is a no-op and must not clobber the original backup
    backup.write_text(json.dumps({"chat_template": TPL, "sentinel": 1}))
    second = sync_bundle_chat_template(b)
    assert second["changed"] is False
    assert json.loads(backup.read_text())["sentinel"] == 1
    assert first["changed"] is True


def test_sync_leaves_jinja_only_bundles_alone(tmp_path):
    """19 of 59 surveyed bundles ship jinja-only ON PURPOSE.

    Adding an embedded copy would create a second source of truth that can
    drift later — exactly the defect this module exists to prevent.
    """
    b = _bundle(tmp_path, jinja=TPL)
    result = sync_bundle_chat_template(b)
    assert result["changed"] is False
    assert "chat_template" not in json.loads((b / "tokenizer_config.json").read_text())


def test_sync_leaves_consistent_bundles_untouched(tmp_path):
    b = _bundle(tmp_path, jinja=TPL, embedded=TPL)
    before = (b / "tokenizer_config.json").read_text()
    assert sync_bundle_chat_template(b)["changed"] is False
    assert (b / "tokenizer_config.json").read_text() == before


def test_dry_run_reports_without_writing(tmp_path):
    b = _bundle(tmp_path, jinja=FIXED, embedded=TPL)
    before = (b / "tokenizer_config.json").read_text()
    result = sync_bundle_chat_template(b, dry_run=True)
    assert result["changed"] is True and result["dry_run"] is True
    assert (b / "tokenizer_config.json").read_text() == before
    assert not (b / "tokenizer_config.json.bak-pre-template-sync").exists()


def test_audit_never_raises_on_a_broken_bundle(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{not json")
    assert audit_bundle_chat_template(tmp_path)["status"] == "unreadable"
    assert sync_bundle_chat_template(tmp_path)["changed"] is False
