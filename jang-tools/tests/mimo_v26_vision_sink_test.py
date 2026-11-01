"""Vision sink semantics and BF16 compatibility, independent of model weights."""
import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from jang_tools.mimo_v2.v26_omni import MiMoV26Omni
from jang_tools.mimo_v2.v26_vision import MiMoV26VisionAttention, MiMoV26VisionConfig


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("length,window", [(3, 64), (7, 1)])
def test_virtual_sink_is_extra_denominator_logit(dtype, length, window):
    cfg = MiMoV26VisionConfig(hidden_size=64, num_heads=1,
                             num_key_value_heads=1, visual_token_window_size=window)
    attn = MiMoV26VisionAttention(cfg, use_sink=True)
    attn.sink_mode = "virtual"
    attn.sinks = mx.array([math.log(2)], dtype=dtype)
    q = mx.zeros((1, 1, length, 64), dtype=dtype)
    v = mx.ones_like(q)
    result = np.array(attn._attend(q, q, v, full_attn=False).astype(mx.float32))
    # Zero q/k gives unit mass to each visible key; the sink contributes
    # exp(log(2)) = 2 to the denominator and zero to the numerator.
    count = np.array([sum(abs(i-j) <= window for j in range(length))
                      for i in range(length)], dtype=np.float32)
    expected = count / (count + 2)
    np.testing.assert_allclose(result[0, 0, :, 0], expected, atol=0.004, rtol=0)
    # A key-0 bias with all-one values would incorrectly return all ones.
    assert float(result.max()) < 0.9


def test_serving_wrapper_selects_virtual_sink_before_loading(monkeypatch, tmp_path):
    from jang_tools.mimo_v2 import v26_omni as omni

    calls = []

    class Tower:
        def __init__(self, cfg, merger_norm):
            pass

        def set_sink_mode(self, mode):
            calls.append(("sink", mode))

        def load_source_weights(self, weights, dtype):
            calls.append(("load", weights))

    monkeypatch.setattr(omni.V.MiMoV26VisionConfig, "from_model_dir", lambda _: None)
    monkeypatch.setattr(omni.V.MiMoV26VisionProcessor, "from_model_dir", lambda _: None)
    monkeypatch.setattr(omni.V, "MiMoVisionTower", Tower)
    monkeypatch.setattr(omni.V, "load_visual_weights_from_source", lambda _: "weights")
    wrapper = MiMoV26Omni(tmp_path, None, None)
    wrapper.vision()
    assert calls == [("sink", "virtual"), ("load", "weights")]
    wrapper.vision()
    assert len(calls) == 2
