import json

import mlx.core as mx
import numpy as np
import pytest

from jang_tools.turboquant.codebook import compute_codebook
from jang_tools.turboquant.pipeline import pack_bits
from jang_tools.turboquant.rotation import generate_random_signs
from jang_tools.turboquant.gather_tq_kernel import gather_tq_matmul
from jang_tools.turboquant.tq_kernel import tq_matmul


def test_mpp_nax_availability_does_not_run_smoke_eval(monkeypatch):
    import jang_tools.turboquant.mpp_nax_kernel as nax_kernel

    nax_kernel.mpp_nax_tensorops_available.cache_clear()

    def fail_eval(*_args, **_kwargs):
        raise AssertionError("availability check must not run a smoke Metal eval")

    monkeypatch.setattr(nax_kernel.mx, "eval", fail_eval)

    assert isinstance(nax_kernel.mpp_nax_tensorops_available(), bool)


def _quantize_rows(weight: np.ndarray, bits: int) -> tuple[mx.array, mx.array, mx.array]:
    out_features, in_features = weight.shape
    codebook = mx.array(compute_codebook(in_features, bits), dtype=mx.float32)
    w = mx.array(weight.astype(np.float32))
    norms = mx.sqrt(mx.sum(w * w, axis=1, keepdims=True))
    norms_safe = mx.maximum(norms, mx.array(1e-8))
    w_normed = w / norms_safe
    boundaries = (codebook[:-1] + codebook[1:]) / 2.0
    indices = mx.zeros(w_normed.shape, dtype=mx.uint8)
    for boundary in boundaries:
        indices = indices + (w_normed > boundary).astype(mx.uint8)
    vals_per_u32 = 32 // bits
    pad = (-in_features) % vals_per_u32
    if pad:
        indices = mx.pad(indices, [(0, 0), (0, pad)])
    packed = pack_bits(indices.reshape(-1), bits).reshape(out_features, -1)
    mx.eval(packed, norms)
    return packed, norms.squeeze(-1).astype(mx.float16), codebook


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("batch", [3, 17])
def test_nax_tq_matmul_matches_existing_kernel(bits, batch):
    from jang_tools.turboquant.mpp_nax_kernel import (
        mpp_nax_tensorops_available,
        tq_matmul_mpp_nax,
    )

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2000 + bits + batch)
    in_features = 64
    out_features = 40
    x = mx.array(rng.standard_normal((batch, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=123), dtype=mx.float32)
    w_rot = rng.standard_normal((out_features, in_features)).astype(np.float32)
    packed, norms, codebook = _quantize_rows(w_rot, bits)

    current = tq_matmul(x, packed, norms, codebook, signs, in_features, bits)
    nax = tq_matmul_mpp_nax(x, packed, norms, codebook, signs, in_features, bits)

    mx.eval(current, nax)
    np.testing.assert_allclose(np.array(nax), np.array(current), rtol=3e-2, atol=5e-2)


def test_tq_matmul_opt_in_uses_nax_path(monkeypatch):
    from jang_tools.turboquant import tq_kernel
    from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2031)
    in_features = 64
    out_features = 16
    bits = 4
    x = mx.array(rng.standard_normal((1, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=55), dtype=mx.float32)
    w_rot = rng.standard_normal((out_features, in_features)).astype(np.float32)
    packed, norms, codebook = _quantize_rows(w_rot, bits)

    called = {"nax": 0}
    real = tq_kernel._tq_matmul_mpp_nax

    def wrapped(*args, **kwargs):
        called["nax"] += 1
        return real(*args, **kwargs)

    monkeypatch.setenv("JANGTQ_MPP_NAX", "1")
    monkeypatch.setattr(tq_kernel, "_tq_matmul_mpp_nax", wrapped)

    y = tq_kernel.tq_matmul(x, packed, norms, codebook, signs, in_features, bits)

    mx.eval(y)
    assert called["nax"] == 1


def test_tq_matmul_auto_uses_nax_only_for_prefill_sized_shapes(monkeypatch):
    from jang_tools.turboquant import tq_kernel

    rng = np.random.default_rng(2035)
    in_features = 64
    out_features = 16
    bits = 4
    signs = mx.ones((in_features,), dtype=mx.float32)
    packed, norms, codebook = _quantize_rows(
        rng.standard_normal((out_features, in_features)).astype(np.float32),
        bits,
    )
    called = {"nax": 0}

    def fake_nax(x, *_args, **_kwargs):
        called["nax"] += 1
        if x.ndim == 1:
            batch = 1
        elif x.ndim > 2:
            batch = int(np.prod(tuple(x.shape[:-1])))
        else:
            batch = int(x.shape[0])
        return mx.zeros((batch, out_features), dtype=mx.float32)

    monkeypatch.setenv("JANGTQ_MPP_NAX", "auto")
    monkeypatch.setattr(tq_kernel, "_tq_matmul_mpp_nax", fake_nax)

    small_x = mx.array(rng.standard_normal((1, in_features)).astype(np.float32))
    small = tq_kernel.tq_matmul(
        small_x, packed, norms, codebook, signs, in_features, bits
    )
    mx.eval(small)
    assert called["nax"] == 0

    large_x = mx.array(rng.standard_normal((512, in_features)).astype(np.float32))
    large = tq_kernel.tq_matmul(
        large_x, packed, norms, codebook, signs, in_features, bits
    )
    mx.eval(large)
    assert called["nax"] == 1


def test_gather_nax_matches_existing_kernel_broadcast_topk():
    from jang_tools.turboquant.mpp_nax_kernel import (
        gather_tq_matmul_mpp_nax_from_rot,
        mpp_nax_tensorops_available,
    )
    from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2037)
    in_features = 64
    out_features = 40
    n_experts = 5
    bits = 4
    top_k = 2
    x = mx.array(rng.standard_normal((3, 1, 1, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=57), dtype=mx.float32)
    weights = rng.standard_normal((n_experts, out_features, in_features)).astype(np.float32)
    codebook = None
    packed_rows = []
    norm_rows = []
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(weights[expert], bits)
        packed_rows.append(packed)
        norm_rows.append(norms)
        codebook = cb
    packed = mx.stack(packed_rows, axis=0)
    norms = mx.stack(norm_rows, axis=0)
    indices = mx.array(np.array([[0, 2], [1, 3], [4, 0]], dtype=np.uint32))

    current = gather_tq_matmul(x, packed, norms, codebook, signs, indices, bits)

    x_flat = x.squeeze(-2).squeeze(-2).reshape(-1, in_features)
    x_rot = hadamard_rotate_metal(x_flat.astype(mx.float32), signs)
    idx_flat = indices.reshape(-1).astype(mx.uint32)
    x_rot_broadcast = mx.repeat(x_rot, top_k, axis=0)
    nax_raw = gather_tq_matmul_mpp_nax_from_rot(
        x_rot_broadcast,
        packed,
        norms,
        codebook,
        idx_flat,
        in_features,
        out_features,
        bits,
    )
    nax = nax_raw.reshape(3, top_k, 1, out_features).astype(current.dtype)

    mx.eval(current, nax)
    np.testing.assert_allclose(np.array(nax), np.array(current), rtol=3e-2, atol=6e-2)


def test_grouped_gather_nax_matches_existing_sorted_kernel():
    from jang_tools.turboquant.mpp_nax_kernel import (
        gather_tq_matmul_mpp_nax_grouped_from_rot,
        mpp_nax_tensorops_available,
    )
    from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2041)
    in_features = 64
    out_features = 40
    n_experts = 4
    bits = 4
    n_dispatches = 37
    x_rot = mx.array(rng.standard_normal((n_dispatches, in_features)).astype(np.float32))
    signs = mx.ones((in_features,), dtype=mx.float32)
    weights = rng.standard_normal((n_experts, out_features, in_features)).astype(np.float32)
    codebook = None
    packed_rows = []
    norm_rows = []
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(weights[expert], bits)
        packed_rows.append(packed)
        norm_rows.append(norms)
        codebook = cb
    packed = mx.stack(packed_rows, axis=0)
    norms = mx.stack(norm_rows, axis=0)
    indices_np = np.array(
        [0] * 5 + [1] * 17 + [2] * 3 + [3] * 12,
        dtype=np.uint32,
    )
    indices = mx.array(indices_np)

    current = gather_tq_matmul(
        x_rot.reshape(n_dispatches, 1, in_features),
        packed,
        norms,
        codebook,
        signs,
        indices,
        bits,
        sorted_indices=True,
    ).reshape(n_dispatches, out_features)
    grouped = gather_tq_matmul_mpp_nax_grouped_from_rot(
        hadamard_rotate_metal(x_rot.astype(mx.float32), signs),
        packed,
        norms,
        codebook,
        indices,
        in_features,
        out_features,
        bits,
    ).astype(current.dtype)

    mx.eval(current, grouped)
    np.testing.assert_allclose(
        np.array(grouped), np.array(current), rtol=3e-2, atol=6e-2
    )


def test_gather_sorted_opt_in_uses_grouped_nax(monkeypatch):
    import jang_tools.turboquant.gather_tq_kernel as gather_kernel
    from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2047)
    in_features = 64
    out_features = 16
    n_experts = 3
    bits = 4
    n_dispatches = 18
    x = mx.array(rng.standard_normal((n_dispatches, 1, in_features)).astype(np.float32))
    signs = mx.ones((in_features,), dtype=mx.float32)
    packed_rows = []
    norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        packed_rows.append(packed)
        norm_rows.append(norms)
        codebook = cb
    packed = mx.stack(packed_rows, axis=0)
    norms = mx.stack(norm_rows, axis=0)
    indices = mx.array(np.array([0] * 6 + [1] * 6 + [2] * 6, dtype=np.uint32))

    called = {"grouped": 0}
    real = gather_kernel._gather_tq_mpp_nax_grouped_from_rot

    def wrapped(*args, **kwargs):
        called["grouped"] += 1
        return real(*args, **kwargs)

    monkeypatch.setenv("JANGTQ_MPP_NAX", "1")
    monkeypatch.setattr(gather_kernel, "_gather_tq_mpp_nax_grouped_from_rot", wrapped)

    out = gather_tq_matmul(
        x,
        packed,
        norms,
        codebook,
        signs,
        indices,
        bits,
        sorted_indices=True,
    )

    mx.eval(out)
    assert called["grouped"] == 1


def test_gather_sorted_auto_uses_grouped_nax_only_for_measured_win_shapes(monkeypatch):
    import jang_tools.turboquant.gather_tq_kernel as gather_kernel
    from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2049)
    in_features = 64
    out_features = 16
    n_experts = 3
    bits = 4
    signs = mx.ones((in_features,), dtype=mx.float32)
    packed_rows = []
    norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        packed_rows.append(packed)
        norm_rows.append(norms)
        codebook = cb
    packed = mx.stack(packed_rows, axis=0)
    norms = mx.stack(norm_rows, axis=0)

    called = {"grouped": 0}
    real = gather_kernel._gather_tq_mpp_nax_grouped_from_rot

    def wrapped(*args, **kwargs):
        called["grouped"] += 1
        return real(*args, **kwargs)

    monkeypatch.setenv("JANGTQ_MPP_NAX", "auto")
    monkeypatch.setattr(gather_kernel, "_gather_tq_mpp_nax_grouped_from_rot", wrapped)

    small_x = mx.array(rng.standard_normal((128, 1, in_features)).astype(np.float32))
    small_indices = mx.array(np.array([0] * 128, dtype=np.uint32))
    small = gather_tq_matmul(
        small_x,
        packed,
        norms,
        codebook,
        signs,
        small_indices,
        bits,
        sorted_indices=True,
    )
    mx.eval(small)
    assert called["grouped"] == 0

    large_x = mx.array(rng.standard_normal((512, 1, in_features)).astype(np.float32))
    large_indices = mx.array(np.array([0] * 512, dtype=np.uint32))
    large = gather_tq_matmul(
        large_x,
        packed,
        norms,
        codebook,
        signs,
        large_indices,
        bits,
        sorted_indices=True,
    )
    mx.eval(large)
    assert called["grouped"] == 1


def test_gather_sorted_prefill_defaults_to_grouped_nax(monkeypatch):
    import jang_tools.turboquant.gather_tq_kernel as gather_kernel

    rng = np.random.default_rng(2050)
    in_features = 64
    out_features = 16
    n_experts = 3
    bits = 4
    signs = mx.ones((in_features,), dtype=mx.float32)
    packed_rows = []
    norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        packed_rows.append(packed)
        norm_rows.append(norms)
        codebook = cb
    packed = mx.stack(packed_rows, axis=0)
    norms = mx.stack(norm_rows, axis=0)
    x = mx.array(rng.standard_normal((512, 1, in_features)).astype(np.float32))
    indices = mx.array(np.array([0] * 512, dtype=np.uint32))

    called = {"grouped": 0}

    def fake_grouped(x_rot, packed_arg, norms_arg, codebook_arg, idx_flat, in_f, out_f, bits_arg):
        called["grouped"] += 1
        assert in_f == in_features
        assert out_f == out_features
        assert bits_arg == bits
        return mx.zeros((idx_flat.size, out_features), dtype=mx.float32)

    monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
    monkeypatch.delenv("JANGTQ_MPP_NAX_PREFILL", raising=False)
    monkeypatch.setattr(gather_kernel, "_gather_tq_mpp_nax_grouped_from_rot", fake_grouped)

    out = gather_tq_matmul(
        x,
        packed,
        norms,
        codebook,
        signs,
        indices,
        bits,
        sorted_indices=True,
    )

    mx.eval(out)
    assert called["grouped"] == 1


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [("JANGTQ_MPP_NAX", "0"), ("JANGTQ_MPP_NAX_PREFILL", "0")],
)
def test_gather_sorted_prefill_can_disable_default_nax(monkeypatch, env_name, env_value):
    import jang_tools.turboquant.gather_tq_kernel as gather_kernel

    rng = np.random.default_rng(2052)
    in_features = 64
    out_features = 16
    n_experts = 3
    bits = 4
    signs = mx.ones((in_features,), dtype=mx.float32)
    packed_rows = []
    norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        packed_rows.append(packed)
        norm_rows.append(norms)
        codebook = cb
    packed = mx.stack(packed_rows, axis=0)
    norms = mx.stack(norm_rows, axis=0)
    x = mx.array(rng.standard_normal((512, 1, in_features)).astype(np.float32))
    indices = mx.array(np.array([0] * 512, dtype=np.uint32))

    called = {"grouped": 0}

    def fake_grouped(*_args, **_kwargs):
        called["grouped"] += 1
        return mx.zeros((512, out_features), dtype=mx.float32)

    monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
    monkeypatch.delenv("JANGTQ_MPP_NAX_PREFILL", raising=False)
    monkeypatch.setenv(env_name, env_value)
    monkeypatch.setattr(gather_kernel, "_gather_tq_mpp_nax_grouped_from_rot", fake_grouped)

    out = gather_tq_matmul(
        x,
        packed,
        norms,
        codebook,
        signs,
        indices,
        bits,
        sorted_indices=True,
    )

    mx.eval(out)
    assert called["grouped"] == 0


def test_sorted_group_tile_metadata_reused_for_same_indices(monkeypatch):
    import jang_tools.turboquant.mpp_nax_kernel as nax_kernel

    indices = mx.array(np.array([0] * 8 + [1] * 8 + [2] * 8, dtype=np.uint32))
    calls = {"build": 0}
    real = nax_kernel._build_sorted_group_tiles_cpu

    def wrapped(idx):
        calls["build"] += 1
        return real(idx)

    monkeypatch.setattr(nax_kernel, "_build_sorted_group_tiles_cpu", wrapped)
    nax_kernel._GROUP_TILE_CACHE["ref"] = None
    nax_kernel._GROUP_TILE_CACHE["tiles"] = None

    first = nax_kernel._build_sorted_group_tiles_cached(indices)
    second = nax_kernel._build_sorted_group_tiles_cached(indices)

    assert calls["build"] == 1
    assert first is second


def test_sorted_cluster_reuses_tile_metadata_across_gate_up_and_down(monkeypatch):
    import jang_tools.turboquant.mpp_nax_kernel as nax_kernel
    from jang_tools.turboquant.fused_gate_up_kernel import fused_gate_up_swiglu_matmul
    from jang_tools.turboquant.gather_tq_kernel import gather_tq_matmul
    from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2049)
    in_features = 64
    intermediate = 40
    n_experts = 4
    bits = 4
    n_dispatches = 32
    signs_in = mx.ones((in_features,), dtype=mx.float32)
    signs_mid = mx.ones((intermediate,), dtype=mx.float32)
    x = mx.array(rng.standard_normal((n_dispatches, 1, in_features)).astype(np.float32))
    indices = mx.array(np.array([0] * 8 + [1] * 8 + [2] * 8 + [3] * 8, dtype=np.uint32))

    def stacked_rows(out_features: int, input_features: int):
        packed_rows = []
        norm_rows = []
        codebook = None
        for _ in range(n_experts):
            packed, norms, cb = _quantize_rows(
                rng.standard_normal((out_features, input_features)).astype(np.float32),
                bits,
            )
            packed_rows.append(packed)
            norm_rows.append(norms)
            codebook = cb
        return mx.stack(packed_rows, axis=0), mx.stack(norm_rows, axis=0), codebook

    gate_packed, gate_norms, codebook = stacked_rows(intermediate, in_features)
    up_packed, up_norms, _ = stacked_rows(intermediate, in_features)
    down_packed, down_norms, down_codebook = stacked_rows(in_features, intermediate)

    calls = {"build": 0}
    real = nax_kernel._build_sorted_group_tiles_cpu

    def wrapped(idx):
        calls["build"] += 1
        return real(idx)

    monkeypatch.setenv("JANGTQ_MPP_NAX", "1")
    monkeypatch.setattr(nax_kernel, "_build_sorted_group_tiles_cpu", wrapped)
    nax_kernel._GROUP_TILE_CACHE["ref"] = None
    nax_kernel._GROUP_TILE_CACHE["tiles"] = None

    act = fused_gate_up_swiglu_matmul(
        x,
        gate_packed,
        gate_norms,
        up_packed,
        up_norms,
        codebook,
        signs_in,
        indices,
        bits,
    )
    out = gather_tq_matmul(
        act,
        down_packed,
        down_norms,
        down_codebook,
        signs_mid,
        indices,
        bits,
        sorted_indices=True,
    )

    mx.eval(out)
    assert calls["build"] == 1


def test_fused_grouped_gate_up_nax_matches_existing_sorted_kernel():
    from jang_tools.turboquant.fused_gate_up_kernel import fused_gate_up_swiglu_matmul
    from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal
    from jang_tools.turboquant.mpp_nax_kernel import (
        fused_gate_up_swiglu_mpp_nax_grouped_from_rot,
        mpp_nax_tensorops_available,
    )

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2051)
    in_features = 64
    out_features = 40
    n_experts = 4
    bits = 4
    n_dispatches = 37
    x = mx.array(rng.standard_normal((n_dispatches, 1, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=51), dtype=mx.float32)
    gate_rows = []
    gate_norm_rows = []
    up_rows = []
    up_norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        gate_rows.append(packed)
        gate_norm_rows.append(norms)
        codebook = cb
        packed, norms, _ = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        up_rows.append(packed)
        up_norm_rows.append(norms)
    packed_gate = mx.stack(gate_rows, axis=0)
    norms_gate = mx.stack(gate_norm_rows, axis=0)
    packed_up = mx.stack(up_rows, axis=0)
    norms_up = mx.stack(up_norm_rows, axis=0)
    indices = mx.array(
        np.array([0] * 5 + [1] * 17 + [2] * 3 + [3] * 12, dtype=np.uint32)
    )

    current = fused_gate_up_swiglu_matmul(
        x,
        packed_gate,
        norms_gate,
        packed_up,
        norms_up,
        codebook,
        signs,
        indices,
        bits,
    ).reshape(n_dispatches, out_features)
    x_rot = hadamard_rotate_metal(x.reshape(n_dispatches, in_features), signs)
    grouped = fused_gate_up_swiglu_mpp_nax_grouped_from_rot(
        x_rot,
        packed_gate,
        norms_gate,
        packed_up,
        norms_up,
        codebook,
        indices,
        in_features,
        out_features,
        bits,
    ).astype(current.dtype)

    mx.eval(current, grouped)
    np.testing.assert_allclose(
        np.array(grouped), np.array(current), rtol=3e-2, atol=7e-2
    )


def test_fused_gate_up_swiglu_opt_in_routes_through_nax(monkeypatch):
    import jang_tools.turboquant.fused_gate_up_kernel as fused_kernel
    from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2053)
    in_features = 64
    out_features = 16
    n_experts = 3
    bits = 4
    batch = 4
    top_k = 2
    x = mx.array(rng.standard_normal((batch, 1, 1, in_features)).astype(np.float32))
    signs = mx.ones((in_features,), dtype=mx.float32)
    packed_rows = []
    norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        packed_rows.append(packed)
        norm_rows.append(norms)
        codebook = cb
    packed = mx.stack(packed_rows, axis=0)
    norms = mx.stack(norm_rows, axis=0)
    indices = mx.array(
        np.array([[0, 1], [1, 2], [2, 0], [0, 2]], dtype=np.uint32)
    )

    called = {"nax": 0}

    def fake_nax(x_rot, packed_arg, norms_arg, codebook_arg, idx_flat, in_f, out_f, bits_arg):
        called["nax"] += 1
        assert in_f == in_features
        assert out_f == out_features
        assert bits_arg == bits
        return mx.ones((idx_flat.size, out_features), dtype=mx.float32)

    monkeypatch.setenv("JANGTQ_MPP_NAX", "1")
    monkeypatch.setattr(
        fused_kernel,
        "_fused_gate_up_swiglu_mpp_nax_from_rot",
        fake_nax,
        raising=False,
    )

    y = fused_kernel.fused_gate_up_swiglu_matmul(
        x,
        packed,
        norms,
        packed,
        norms,
        codebook,
        signs,
        indices,
        bits,
    )

    mx.eval(y)
    assert called["nax"] == 2
    assert y.shape == (batch, top_k, 1, out_features)


def test_fused_gate_up_swiglu_sorted_opt_in_routes_through_grouped_nax(monkeypatch):
    import jang_tools.turboquant.fused_gate_up_kernel as fused_kernel
    from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2055)
    in_features = 64
    out_features = 16
    n_experts = 3
    bits = 4
    n_dispatches = 18
    x = mx.array(rng.standard_normal((n_dispatches, 1, in_features)).astype(np.float32))
    signs = mx.ones((in_features,), dtype=mx.float32)
    packed_rows = []
    norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        packed_rows.append(packed)
        norm_rows.append(norms)
        codebook = cb
    packed = mx.stack(packed_rows, axis=0)
    norms = mx.stack(norm_rows, axis=0)
    indices = mx.array(np.array([0] * 6 + [1] * 6 + [2] * 6, dtype=np.uint32))

    called = {"grouped": 0}

    def fake_grouped(
        x_rot,
        packed_gate,
        norms_gate,
        packed_up,
        norms_up,
        codebook_arg,
        idx_flat,
        in_f,
        out_f,
        bits_arg,
        swiglu_limit=0.0,
    ):
        called["grouped"] += 1
        assert in_f == in_features
        assert out_f == out_features
        assert bits_arg == bits
        return mx.ones((idx_flat.size, out_features), dtype=mx.float32)

    monkeypatch.setenv("JANGTQ_MPP_NAX", "1")
    monkeypatch.setattr(
        fused_kernel,
        "_fused_gate_up_swiglu_mpp_nax_grouped_from_rot",
        fake_grouped,
        raising=False,
    )

    y = fused_kernel.fused_gate_up_swiglu_matmul(
        x,
        packed,
        norms,
        packed,
        norms,
        codebook,
        signs,
        indices,
        bits,
    )

    mx.eval(y)
    assert called["grouped"] == 1
    assert y.shape == (n_dispatches, 1, out_features)


def test_fused_gate_up_swiglu_auto_uses_grouped_nax_only_for_measured_win_shapes(monkeypatch):
    import jang_tools.turboquant.fused_gate_up_kernel as fused_kernel
    from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2057)
    in_features = 64
    out_features = 16
    n_experts = 3
    bits = 4
    signs = mx.ones((in_features,), dtype=mx.float32)
    packed_rows = []
    norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        packed_rows.append(packed)
        norm_rows.append(norms)
        codebook = cb
    packed = mx.stack(packed_rows, axis=0)
    norms = mx.stack(norm_rows, axis=0)

    called = {"grouped": 0}

    def fake_grouped(
        x_rot,
        packed_gate,
        norms_gate,
        packed_up,
        norms_up,
        codebook_arg,
        idx_flat,
        in_f,
        out_f,
        bits_arg,
        swiglu_limit=0.0,
    ):
        called["grouped"] += 1
        return mx.ones((idx_flat.size, out_features), dtype=mx.float32)

    monkeypatch.setenv("JANGTQ_MPP_NAX", "auto")
    monkeypatch.setattr(
        fused_kernel,
        "_fused_gate_up_swiglu_mpp_nax_grouped_from_rot",
        fake_grouped,
        raising=False,
    )

    small_x = mx.array(rng.standard_normal((128, 1, in_features)).astype(np.float32))
    small_indices = mx.array(np.array([0] * 128, dtype=np.uint32))
    small = fused_kernel.fused_gate_up_swiglu_matmul(
        small_x, packed, norms, packed, norms, codebook, signs, small_indices, bits
    )
    mx.eval(small)
    assert called["grouped"] == 0

    large_x = mx.array(rng.standard_normal((512, 1, in_features)).astype(np.float32))
    large_indices = mx.array(np.array([0] * 512, dtype=np.uint32))
    large = fused_kernel.fused_gate_up_swiglu_matmul(
        large_x, packed, norms, packed, norms, codebook, signs, large_indices, bits
    )
    mx.eval(large)
    assert called["grouped"] == 1


def test_fused_gate_up_swiglu_sorted_prefill_defaults_to_grouped_nax(monkeypatch):
    import jang_tools.turboquant.fused_gate_up_kernel as fused_kernel

    rng = np.random.default_rng(2058)
    in_features = 64
    out_features = 16
    n_experts = 3
    bits = 4
    signs = mx.ones((in_features,), dtype=mx.float32)
    packed_rows = []
    norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        packed_rows.append(packed)
        norm_rows.append(norms)
        codebook = cb
    packed = mx.stack(packed_rows, axis=0)
    norms = mx.stack(norm_rows, axis=0)
    x = mx.array(rng.standard_normal((512, 1, in_features)).astype(np.float32))
    indices = mx.array(np.array([0] * 512, dtype=np.uint32))

    called = {"grouped": 0}

    def fake_grouped(
        x_rot,
        packed_gate,
        norms_gate,
        packed_up,
        norms_up,
        codebook_arg,
        idx_flat,
        in_f,
        out_f,
        bits_arg,
        swiglu_limit=0.0,
    ):
        called["grouped"] += 1
        assert in_f == in_features
        assert out_f == out_features
        assert bits_arg == bits
        return mx.zeros((idx_flat.size, out_features), dtype=mx.float32)

    monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
    monkeypatch.delenv("JANGTQ_MPP_NAX_PREFILL", raising=False)
    monkeypatch.setattr(
        fused_kernel,
        "_fused_gate_up_swiglu_mpp_nax_grouped_from_rot",
        fake_grouped,
        raising=False,
    )

    y = fused_kernel.fused_gate_up_swiglu_matmul(
        x,
        packed,
        norms,
        packed,
        norms,
        codebook,
        signs,
        indices,
        bits,
    )

    mx.eval(y)
    assert called["grouped"] == 1


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [("JANGTQ_MPP_NAX", "0"), ("JANGTQ_MPP_NAX_PREFILL", "0")],
)
def test_fused_gate_up_swiglu_sorted_prefill_can_disable_default_nax(
    monkeypatch,
    env_name,
    env_value,
):
    import jang_tools.turboquant.fused_gate_up_kernel as fused_kernel

    rng = np.random.default_rng(2060)
    in_features = 64
    out_features = 16
    n_experts = 3
    bits = 4
    signs = mx.ones((in_features,), dtype=mx.float32)
    packed_rows = []
    norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        packed_rows.append(packed)
        norm_rows.append(norms)
        codebook = cb
    packed = mx.stack(packed_rows, axis=0)
    norms = mx.stack(norm_rows, axis=0)
    x = mx.array(rng.standard_normal((512, 1, in_features)).astype(np.float32))
    indices = mx.array(np.array([0] * 512, dtype=np.uint32))

    called = {"grouped": 0}

    def fake_grouped(*_args, **_kwargs):
        called["grouped"] += 1
        return mx.zeros((512, out_features), dtype=mx.float32)

    monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
    monkeypatch.delenv("JANGTQ_MPP_NAX_PREFILL", raising=False)
    monkeypatch.setenv(env_name, env_value)
    monkeypatch.setattr(
        fused_kernel,
        "_fused_gate_up_swiglu_mpp_nax_grouped_from_rot",
        fake_grouped,
        raising=False,
    )

    y = fused_kernel.fused_gate_up_swiglu_matmul(
        x,
        packed,
        norms,
        packed,
        norms,
        codebook,
        signs,
        indices,
        bits,
    )

    mx.eval(y)
    assert called["grouped"] == 0


@pytest.mark.parametrize("swiglu_limit", [0.0, 10.0])
def test_fused_gate_up_swiglu_nax_matches_existing_kernel(monkeypatch, swiglu_limit):
    import jang_tools.turboquant.fused_gate_up_kernel as fused_kernel
    from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2059 + int(swiglu_limit))
    in_features = 64
    out_features = 40
    n_experts = 4
    bits = 4
    batch = 5
    x = mx.array(rng.standard_normal((batch, 1, 1, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=59), dtype=mx.float32)
    gate_rows = []
    gate_norm_rows = []
    up_rows = []
    up_norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        gate_rows.append(packed)
        gate_norm_rows.append(norms)
        codebook = cb
        packed, norms, _ = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        up_rows.append(packed)
        up_norm_rows.append(norms)
    packed_gate = mx.stack(gate_rows, axis=0)
    norms_gate = mx.stack(gate_norm_rows, axis=0)
    packed_up = mx.stack(up_rows, axis=0)
    norms_up = mx.stack(up_norm_rows, axis=0)
    indices = mx.array(
        np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.uint32)
    )

    monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
    current = fused_kernel.fused_gate_up_swiglu_matmul(
        x,
        packed_gate,
        norms_gate,
        packed_up,
        norms_up,
        codebook,
        signs,
        indices,
        bits,
        swiglu_limit=swiglu_limit,
    )
    monkeypatch.setenv("JANGTQ_MPP_NAX", "1")
    nax = fused_kernel.fused_gate_up_swiglu_matmul(
        x,
        packed_gate,
        norms_gate,
        packed_up,
        norms_up,
        codebook,
        signs,
        indices,
        bits,
        swiglu_limit=swiglu_limit,
    )

    mx.eval(current, nax)
    np.testing.assert_allclose(np.array(nax), np.array(current), rtol=3e-2, atol=7e-2)


def test_grouped_fused_gate_up_swiglu_nax_matches_existing_sorted_kernel(monkeypatch):
    import jang_tools.turboquant.fused_gate_up_kernel as fused_kernel
    from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2063)
    in_features = 64
    out_features = 40
    n_experts = 4
    bits = 4
    n_dispatches = 37
    x = mx.array(rng.standard_normal((n_dispatches, 1, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=63), dtype=mx.float32)
    gate_rows = []
    gate_norm_rows = []
    up_rows = []
    up_norm_rows = []
    codebook = None
    for expert in range(n_experts):
        packed, norms, cb = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        gate_rows.append(packed)
        gate_norm_rows.append(norms)
        codebook = cb
        packed, norms, _ = _quantize_rows(
            rng.standard_normal((out_features, in_features)).astype(np.float32),
            bits,
        )
        up_rows.append(packed)
        up_norm_rows.append(norms)
    packed_gate = mx.stack(gate_rows, axis=0)
    norms_gate = mx.stack(gate_norm_rows, axis=0)
    packed_up = mx.stack(up_rows, axis=0)
    norms_up = mx.stack(up_norm_rows, axis=0)
    indices = mx.array(
        np.array([0] * 5 + [1] * 17 + [2] * 3 + [3] * 12, dtype=np.uint32)
    )

    monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
    current = fused_kernel.fused_gate_up_swiglu_matmul(
        x,
        packed_gate,
        norms_gate,
        packed_up,
        norms_up,
        codebook,
        signs,
        indices,
        bits,
        swiglu_limit=10.0,
    )
    monkeypatch.setenv("JANGTQ_MPP_NAX", "1")
    grouped = fused_kernel.fused_gate_up_swiglu_matmul(
        x,
        packed_gate,
        norms_gate,
        packed_up,
        norms_up,
        codebook,
        signs,
        indices,
        bits,
        swiglu_limit=10.0,
    )

    mx.eval(current, grouped)
    np.testing.assert_allclose(
        np.array(grouped), np.array(current), rtol=3e-2, atol=7e-2
    )


def test_full_expert_cluster_nax_matches_existing_broadcast_topk(monkeypatch):
    import jang_tools.turboquant.fused_gate_up_kernel as fused_kernel
    from jang_tools.turboquant.gather_tq_kernel import gather_tq_matmul
    from jang_tools.turboquant.mpp_nax_kernel import mpp_nax_tensorops_available

    if not mpp_nax_tensorops_available():
        pytest.skip("MPP NAX tensor_ops unavailable")

    rng = np.random.default_rng(2067)
    tokens = 32
    top_k = 2
    hidden = 64
    intermediate = 64
    n_experts = 4
    bits = 4
    x = mx.array(rng.standard_normal((tokens, 1, 1, hidden)).astype(np.float32))
    gate_signs = mx.array(generate_random_signs(hidden, seed=67), dtype=mx.float32)
    down_signs = mx.array(
        generate_random_signs(intermediate, seed=68), dtype=mx.float32
    )

    def expert_weights(out_features, in_features):
        packed_rows = []
        norm_rows = []
        codebook = None
        for _ in range(n_experts):
            packed, norms, cb = _quantize_rows(
                rng.standard_normal((out_features, in_features)).astype(np.float32),
                bits,
            )
            packed_rows.append(packed)
            norm_rows.append(norms)
            codebook = cb
        return mx.stack(packed_rows, axis=0), mx.stack(norm_rows, axis=0), codebook

    gate_packed, gate_norms, gate_codebook = expert_weights(intermediate, hidden)
    up_packed, up_norms, _ = expert_weights(intermediate, hidden)
    down_packed, down_norms, down_codebook = expert_weights(hidden, intermediate)
    indices = mx.array(
        rng.integers(0, n_experts, size=(tokens, top_k), dtype=np.uint32)
    )

    def cluster():
        x_act = fused_kernel.fused_gate_up_swiglu_matmul(
            x,
            gate_packed,
            gate_norms,
            up_packed,
            up_norms,
            gate_codebook,
            gate_signs,
            indices,
            bits,
        )
        return gather_tq_matmul(
            x_act,
            down_packed,
            down_norms,
            down_codebook,
            down_signs,
            indices,
            bits,
        )

    monkeypatch.delenv("JANGTQ_MPP_NAX", raising=False)
    current = cluster()
    monkeypatch.setenv("JANGTQ_MPP_NAX", "1")
    nax = cluster()

    mx.eval(current, nax)
    np.testing.assert_allclose(np.array(nax), np.array(current), rtol=4e-2, atol=8e-2)


def test_minimax_kernel_compare_detects_jangtq_metadata(tmp_path):
    from jang_tools.minimax_kernel_compare import detect_model_kind, model_summary

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minimax_m2",
                "weight_format": "mxtq",
                "hidden_size": 3072,
                "intermediate_size": 1536,
                "num_hidden_layers": 62,
                "num_local_experts": 160,
                "num_experts_per_tok": 8,
            }
        )
    )
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"profile": "JANGTQ2", "mxtq_bits": {"routed": 2}})
    )

    assert detect_model_kind(tmp_path) == "jangtq"
    summary = model_summary(tmp_path)
    assert summary["kind"] == "jangtq"
    assert summary["profile"] == "JANGTQ2"
    assert summary["weight_format"] == "mxtq"
    assert summary["hidden_size"] == 3072
    assert summary["intermediate_size"] == 1536
    assert summary["num_local_experts"] == 160
    assert summary["num_experts_per_tok"] == 8


def test_minimax_kernel_compare_detects_affine_jang_metadata(tmp_path):
    from jang_tools.minimax_kernel_compare import detect_model_kind, model_summary

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minimax_m2",
                "hidden_size": 3072,
                "intermediate_size": 1536,
                "num_hidden_layers": 62,
                "num_local_experts": 256,
                "num_experts_per_tok": 8,
                "quantization": {"bits": 2, "group_size": 64, "mode": "affine"},
            }
        )
    )
    (tmp_path / "jang_config.json").write_text(
        json.dumps(
            {
                "format": "jang-v2",
                "quantization": {
                    "method": "jang-importance",
                    "profile": "JANG_2L",
                    "target_bits": 2.0,
                    "actual_bits": 2.1,
                },
            }
        )
    )

    assert detect_model_kind(tmp_path) == "jang"
    summary = model_summary(tmp_path)
    assert summary["kind"] == "jang"
    assert summary["profile"] == "JANG_2L"
    assert summary["weight_format"] == "affine"
    assert summary["quantization_bits"] == 2
    assert summary["quantization_group_size"] == 64


def test_minimax_kernel_compare_modes_and_env_are_explicit():
    from jang_tools.minimax_kernel_compare import env_for_mode, mode_labels

    assert mode_labels("jangtq", include_global_auto=True) == [
        "legacy_prefill",
        "default_prefill",
        "global_auto",
    ]
    assert mode_labels("jang") == ["affine_default"]

    legacy = env_for_mode("legacy_prefill")
    assert legacy["JANGTQ_MPP_NAX_PREFILL"] == "0"
    assert legacy["JANGTQ_MPP_NAX"] is None
    assert legacy["JANGTQ_MPP_NAX_STRICT"] is None

    default = env_for_mode("default_prefill")
    assert default["JANGTQ_MPP_NAX_PREFILL"] is None
    assert default["JANGTQ_MPP_NAX"] is None
    assert default["JANGTQ_MPP_NAX_STRICT"] == "1"

    affine = env_for_mode("affine_default")
    assert affine == {}


def test_fix_quantized_bits_prefers_jang_block_size_for_ambiguous_switch_shapes():
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    from jang_tools.loader import _fix_quantized_bits

    switch = QuantizedSwitchLinear(
        input_dims=768,
        output_dims=8,
        num_experts=2,
        group_size=32,
        bits=8,
        bias=False,
    )
    assert switch.weight.shape == (2, 8, 192)
    assert switch.scales.shape == (2, 8, 24)

    class Model:
        def named_modules(self):
            yield "model.layers.0.block_sparse_moe.switch_mlp.up_proj", switch

    _fix_quantized_bits(Model(), {}, preferred_group_size=128)

    assert switch.bits == 2
    assert switch.group_size == 128
