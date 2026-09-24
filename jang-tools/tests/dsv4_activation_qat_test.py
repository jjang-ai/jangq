import inspect
from types import SimpleNamespace

import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _run_qat_contract_on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _np(value):
    if value.dtype == mx.bfloat16:
        value = value.astype(mx.float32)
    return np.asarray(value)


@pytest.fixture
def _qat_status_guard():
    import jang_tools.dsv4.mlx_model as model

    previous = dict(model._DSV4_ACTIVATION_QAT_STATUS)
    try:
        yield model._DSV4_ACTIVATION_QAT_STATUS
    finally:
        model._DSV4_ACTIVATION_QAT_STATUS.clear()
        model._DSV4_ACTIVATION_QAT_STATUS.update(previous)


def _set_qat_state(status, enabled):
    status.update(
        requested=enabled,
        effective=enabled,
        observed=None,
        attested=False,
        e4m3_kv_pool_observed=None,
        hadamard_fp4_indexer_observed=None,
    )


def test_dsv4_activation_qat_env_is_explicit_opt_in_and_invalid_is_off():
    from jang_tools.dsv4.mlx_model import _dsv4_activation_qat_requested

    assert _dsv4_activation_qat_requested({}) is False
    for value in ("1", "true", "TRUE", " yes ", "On"):
        assert _dsv4_activation_qat_requested(
            {"DSV4_ACTIVATION_QAT": value}
        ) is True
    for value in ("", "0", "false", "no", "off", "enabled", "garbage"):
        assert _dsv4_activation_qat_requested(
            {"DSV4_ACTIVATION_QAT": value}
        ) is False


def test_dsv4_activation_qat_off_is_identity_and_attests_both_paths(
    _qat_status_guard,
):
    import jang_tools.dsv4.mlx_model as model

    _set_qat_state(_qat_status_guard, False)
    kv = mx.arange(128, dtype=mx.float32).reshape(1, 1, 128)
    indexer = mx.arange(128, dtype=mx.float32).reshape(1, 1, 128)

    assert model._fp8_qat_non_rope(kv, rope_dims=64) is kv
    assert model._indexer_activation_roundtrip(indexer) is indexer
    assert _qat_status_guard["attested"] is True
    assert _qat_status_guard["observed"] is False
    assert _qat_status_guard["e4m3_kv_pool_observed"] is False
    assert _qat_status_guard["hadamard_fp4_indexer_observed"] is False


def test_dsv4_activation_qat_on_matches_existing_source_native_ops(
    _qat_status_guard,
):
    import jang_tools.dsv4.mlx_model as model

    _set_qat_state(_qat_status_guard, True)
    kv = mx.linspace(-511.0, 511.0, 256, dtype=mx.float32).reshape(
        1, 2, 128
    )
    indexer = mx.linspace(-9.0, 9.0, 256, dtype=mx.float32).reshape(
        1, 2, 128
    )

    actual_kv = model._fp8_qat_non_rope(kv, rope_dims=64)
    expected_kv = model._fp8_qat_non_rope_ops(kv, rope_dims=64)
    actual_indexer = model._indexer_activation_roundtrip(indexer)
    expected_indexer = model._indexer_activation_roundtrip_ops(indexer)
    mx.eval(actual_kv, expected_kv, actual_indexer, expected_indexer)

    np.testing.assert_array_equal(_np(actual_kv), _np(expected_kv))
    np.testing.assert_array_equal(_np(actual_indexer), _np(expected_indexer))
    assert _qat_status_guard["attested"] is True
    assert _qat_status_guard["observed"] is True
    assert _qat_status_guard["e4m3_kv_pool_observed"] is True
    assert _qat_status_guard["hadamard_fp4_indexer_observed"] is True


def test_dsv4_model_exposes_live_activation_qat_status(_qat_status_guard):
    from jang_tools.dsv4.mlx_model import Model, ModelArgs

    runtime_model = Model(
        ModelArgs(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=0,
            hc_mult=1,
        )
    )

    assert runtime_model._vmlx_dsv4_activation_qat_status is _qat_status_guard
    assert _qat_status_guard["fp32_compressor_staging_unconditional"] is True
    assert _qat_status_guard["attestation_scope"] == (
        "transform_family_dispatch_not_every_call_site"
    )
    assert _qat_status_guard["transform_families"] == [
        "e4m3_post_rope_kv_or_compressor_pool_dispatch",
        "hadamard_fp4_indexer_pool_or_query_dispatch",
    ]


def test_dsv4_e4m3_block_qat_matches_reference_value_lattice():
    from jang_tools.dsv4.mlx_model import act_quant_sim

    # A maximum of 448 makes the block's UE8M0 scale exactly 1.  These values
    # cover subnormal, normal, tie-to-even, saturation, and signed rounding.
    values = np.array(
        [
            0.0,
            0.001,
            0.001953125,
            0.0029296875,
            0.0146484375,
            0.96875,
            1.0625,
            1.1875,
            432.0,
            448.0,
            -0.0029296875,
            -1.1875,
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            0.0,
            0.001953125,
            0.001953125,
            0.00390625,
            0.015625,
            1.0,
            1.0,
            1.25,
            448.0,
            448.0,
            -0.00390625,
            -1.25,
        ],
        dtype=np.float32,
    )
    block = np.pad(values, (0, 64 - values.size), constant_values=448.0)

    actual = act_quant_sim(mx.array(block), block_size=64)
    mx.eval(actual)

    np.testing.assert_array_equal(_np(actual)[: values.size], expected)


def test_dsv4_fp4_block_qat_matches_e2m1_ties_and_scale():
    from jang_tools.dsv4.mlx_model import fp4_act_quant_sim

    # A maximum of 6 makes the power-of-two block scale exactly 1.
    values = np.array(
        [
            0.0,
            0.25,
            0.2501,
            0.75,
            1.25,
            1.2501,
            1.75,
            2.5,
            2.5001,
            3.5,
            5.0,
            5.0001,
            6.0,
            -0.75,
            -5.0,
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            0.0,
            0.0,
            0.5,
            1.0,
            1.0,
            1.5,
            2.0,
            2.0,
            3.0,
            4.0,
            4.0,
            6.0,
            6.0,
            -1.0,
            -4.0,
        ],
        dtype=np.float32,
    )
    block = np.pad(values, (0, 32 - values.size), constant_values=6.0)

    actual = fp4_act_quant_sim(mx.array(block), block_size=32)
    mx.eval(actual)

    np.testing.assert_array_equal(_np(actual)[: values.size], expected)


def test_dsv4_indexer_hadamard_is_normalized_sylvester_transform():
    from jang_tools.dsv4.mlx_model import hadamard_rotate_activation

    actual = hadamard_rotate_activation(mx.array([[1.0, 2.0, 3.0, 4.0]]))
    mx.eval(actual)

    np.testing.assert_array_equal(_np(actual), [[5.0, -1.0, -2.0, 0.0]])


def test_dsv4_qat_is_wired_after_rope_and_before_cache_or_scoring():
    from jang_tools.dsv4.mlx_model import Compressor, DeepseekV4Attention, Indexer

    attention_source = inspect.getsource(DeepseekV4Attention.__call__)
    kv_rope = attention_source.index("kv = _apply_partial_rope")
    kv_qat = attention_source.index("kv = _fp8_qat_non_rope")
    cache_update = attention_source.index("local_cache.update_and_fetch")
    assert kv_rope < kv_qat < cache_update

    compressor_source = inspect.getsource(Compressor._advance)
    pooled_rope = compressor_source.index("new_pooled = _apply_partial_rope")
    indexer_qat = compressor_source.index("_indexer_activation_roundtrip")
    main_fp8 = compressor_source.index("_fp8_qat_non_rope")
    cache_pool_update = compressor_source.index("cache.update_pool_view")
    assert "_indexer_activation_roundtrip(new_pooled)" in compressor_source
    assert pooled_rope < indexer_qat < cache_pool_update
    assert pooled_rope < main_fp8 < cache_pool_update

    indexer_source = inspect.getsource(Indexer.select)
    q_rope = indexer_source.index("q = _apply_partial_rope")
    q_qat = indexer_source.index("_indexer_activation_roundtrip")
    scoring = indexer_source.index("scores = _dsv4_indexer_scores")
    assert "_indexer_activation_roundtrip(q)" in indexer_source
    assert q_rope < q_qat < scoring

    cfg = SimpleNamespace(
        hidden_size=16,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
        index_n_heads=1,
        index_head_dim=128,
        index_topk=4,
        q_lora_rank=64,
    )
    indexer = Indexer(cfg, compress_ratio=4)
    assert indexer.compressor.rotate is True


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_dsv4_fused_e4m3_kv_qat_matches_pure_ops_and_preserves_rope(dtype):
    import jang_tools.dsv4.mlx_model as model

    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    mx.set_default_device(mx.gpu)
    kernel = model._make_e4m3_kv_activation_roundtrip_kernel()
    assert kernel is not None

    values = np.linspace(-511.0, 511.0, num=2 * 3 * 512, dtype=np.float32)
    values = values.reshape(2, 3, 512)
    x = mx.array(values).astype(dtype)
    expected = model._fp8_qat_non_rope_ops(x, rope_dims=64)
    contiguous = mx.contiguous(x)
    actual = kernel(
        inputs=[contiguous],
        template=[
            ("N", 512),
            ("NBQ", 7),
            ("NBT", 8),
            ("outT", dtype),
        ],
        grid=(int(x.size), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[dtype],
    )[0]
    mx.eval(actual, expected)

    np.testing.assert_array_equal(_np(actual), _np(expected))
    np.testing.assert_array_equal(_np(actual[..., -64:]), _np(x[..., -64:]))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_dsv4_fused_indexer_qat_matches_pure_ops(dtype):
    import jang_tools.dsv4.mlx_model as model

    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    mx.set_default_device(mx.gpu)
    kernel = model._make_indexer_activation_roundtrip_kernel()
    assert kernel is not None

    values = np.linspace(-9.0, 9.0, num=2 * 3 * 128, dtype=np.float32)
    x = mx.array(values.reshape(2, 3, 128)).astype(dtype)
    expected = model._indexer_activation_roundtrip_ops(x)
    contiguous = mx.contiguous(x)
    actual = kernel(
        inputs=[contiguous],
        template=[("outT", dtype)],
        grid=(int(x.size), 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[dtype],
    )[0]
    mx.eval(actual, expected)

    np.testing.assert_array_equal(_np(actual), _np(expected))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_dsv4_fused_hc_decode_matches_source_order_ops(dtype, monkeypatch):
    import jang_tools.dsv4.mlx_model as model

    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    mx.set_default_device(mx.gpu)
    kernel = model._make_hc_post_decode_kernel()
    assert kernel is not None
    monkeypatch.setattr(model, "_hc_post_decode_kernel", kernel)

    rng = np.random.default_rng(731)
    x = mx.array(rng.normal(size=(1, 1, 64)).astype(np.float32)).astype(dtype)
    residual = mx.array(
        rng.normal(size=(1, 1, 4, 64)).astype(np.float32)
    ).astype(dtype)
    post = mx.array(rng.normal(size=(1, 1, 4)).astype(np.float32))
    comb = mx.array(rng.normal(size=(1, 1, 4, 4)).astype(np.float32))

    expected = model._dsv4_hc_post_ops(x, residual, post, comb)
    actual = model._dsv4_hc_post(x, residual, post, comb)
    mx.eval(actual, expected)

    assert actual.dtype == dtype
    np.testing.assert_array_equal(_np(actual), _np(expected))


def test_dsv4_hc_prefill_retains_source_order_fallback(monkeypatch):
    import jang_tools.dsv4.mlx_model as model

    class DecodeOnlyKernel:
        def __call__(self, *args, **kwargs):
            raise AssertionError("multi-token prefill must not use decode kernel")

    monkeypatch.setattr(model, "_hc_post_decode_kernel", DecodeOnlyKernel())
    x = mx.ones((1, 3, 16), dtype=mx.float16)
    residual = mx.ones((1, 3, 4, 16), dtype=mx.float16)
    post = mx.ones((1, 3, 4), dtype=mx.float32)
    comb = mx.ones((1, 3, 4, 4), dtype=mx.float32)

    expected = model._dsv4_hc_post_ops(x, residual, post, comb)
    actual = model._dsv4_hc_post(x, residual, post, comb)
    mx.eval(actual, expected)

    np.testing.assert_array_equal(_np(actual), _np(expected))


def test_dsv4_folded_fp32_norm_does_not_promote_activation_stream():
    import mlx.nn as nn
    from jang_tools.dsv4.mlx_model import (
        _dsv4_norm_preserve_activation_dtype,
    )

    norm = nn.RMSNorm(16, eps=1e-6)
    norm.weight = mx.ones((16,), dtype=mx.float32)
    activation = mx.arange(16, dtype=mx.float32).reshape(1, 1, 16).astype(
        mx.float16
    )

    promoted = norm(activation)
    actual = _dsv4_norm_preserve_activation_dtype(norm, activation)
    mx.eval(promoted, actual)

    assert promoted.dtype == mx.float32
    assert actual.dtype == mx.float16
    np.testing.assert_array_equal(_np(actual), _np(promoted.astype(mx.float16)))


def test_dsv4_compressor_stages_q8_projection_and_pooling_in_fp32():
    from jang_tools.dsv4.mlx_model import Compressor

    events = []

    class RecordingProjection:
        def __init__(self, inner, name):
            self.inner = inner
            self.name = name

        def __call__(self, value):
            events.append((self.name, "input", value.dtype))
            output = self.inner(value)
            events.append((self.name, "output", output.dtype))
            return output

    class RecordingNorm:
        def __call__(self, value):
            events.append(("norm", "input", value.dtype))
            return value

    class IdentityRope:
        dims = 64
        inv_freq = mx.zeros((32,), dtype=mx.float32)

        def __call__(self, value, offset=0, inverse=False, positions=None):
            return value

    cfg = SimpleNamespace(
        hidden_size=64,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
    )
    compressor = Compressor(cfg, compress_ratio=4, head_dim=128)
    for name in ("wkv", "wgate"):
        quantized = getattr(compressor, name).to_quantized(
            group_size=64,
            bits=8,
        )
        # Match the exact 0731 JANG artifact's affine sidecar dtype.
        quantized.scales = quantized.scales.astype(mx.float16)
        quantized.biases = quantized.biases.astype(mx.float16)
        setattr(compressor, name, RecordingProjection(quantized, name))
    compressor.norm = RecordingNorm()

    output = compressor(
        mx.ones((1, 4, 64), dtype=mx.float16),
        IdentityRope(),
        cache=None,
        start_pos=0,
    )
    mx.eval(output)

    assert ("wkv", "input", mx.float32) in events
    assert ("wkv", "output", mx.float32) in events
    assert ("wgate", "input", mx.float32) in events
    assert ("wgate", "output", mx.float32) in events
    assert ("norm", "input", mx.float16) in events
    assert output.dtype == mx.float16
