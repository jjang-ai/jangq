import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "_internal/jang_v3").exists(),
    reason="_internal/jang_v3 helpers are private and not shipped in the public repo",
)


def test_budget_solver_never_upgrades_dsv4_routed_layers_to_affine_8bit():
    """DSV4 V3 routed layers are only production-proven as 2/4-bit MXTQ."""
    from _internal.jang_v3 import budget_solver

    groups = {
        "model.layers.10.mlp.switch_mlp.SWITCH_MLP_LAYER": {
            "shape_per_unit": [256, 256],
            "n_units": 256,
            "disk_names": [],
        },
    }
    importance = {
        "model.layers.10.mlp.switch_mlp.SWITCH_MLP_LAYER": 1_000_000.0,
    }

    plan = budget_solver.solve(
        groups,
        importance,
        budget_bytes=10**12,
        start_bits=2,
        tiers=[2, 4, 8],
    )

    assert plan["model.layers.10.mlp.switch_mlp.SWITCH_MLP_LAYER"] == 4


def test_v3_encode_lookup_rejects_affine_routed_dsv4_plan(monkeypatch, tmp_path):
    """The dormant lookup helper must match the canonical converter's safety rule."""
    from _internal.jang_v3 import encode

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "plan": {
            "model.layers.9.mlp.switch_mlp.SWITCH_MLP_LAYER": 8,
        },
        "config": {},
    }))
    monkeypatch.setenv("DSV4_V3_PLAN_PATH", str(plan_path))
    encode._PLAN_CACHE = None
    encode._CONFIG_CACHE = None

    try:
        encode.lookup("layers.9.ffn.experts.0.w1.weight")
    except ValueError as exc:
        assert "affine routed" in str(exc)
    else:
        raise AssertionError("expected affine-routed V3 plan to be rejected")


def test_v3_encode_cli_invokes_canonical_converter_v3_variant():
    """The helper CLI must not set a plan env while forgetting --variant V3."""
    src = (Path(__file__).resolve().parents[1] / "_internal/jang_v3/encode.py").read_text()

    assert '"--profile"' in src
    assert '"--profile-bits"' not in src
    assert '"--variant", "V3"' in src
