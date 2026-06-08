from pathlib import Path


def test_load_jangtq_recognizes_prestacked_nemotron_h_switch_mlp_tensors():
    src = (Path(__file__).resolve().parents[1] / "jang_tools" / "load_jangtq.py").read_text()

    assert "nemo_prestack_pat" in src
    assert "switch_mlp\\.(fc1|fc2)" in src
    assert 'proj_name = m.group(2)' in src


def test_skip_params_eval_skips_jangtq_warmup_for_large_smoke_loads():
    src = (Path(__file__).resolve().parents[1] / "jang_tools" / "load_jangtq.py").read_text()

    assert "if skip_params_eval:" in src
    assert "[warmup] skipped because skip_params_eval=True" in src


def test_load_jangtq_patches_nemotron_h_switch_mlp_fast_path():
    src = (Path(__file__).resolve().parents[1] / "jang_tools" / "load_jangtq.py").read_text()

    assert "SwitchMLP" in src
    assert "_fused_switchmlp_call" in src
    assert "Patched SwitchMLP class for fused fc1+relu2+fc2" in src
    assert "make_gather_tq_decode_broadcast" in src
    assert "JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH" in src
    assert "JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH" in src
    assert "mx.repeat(x_rot, k, axis=0)" not in src


def test_gather_kernel_exposes_broadcast_decode_helper_for_nemotron_fc1():
    src = (Path(__file__).resolve().parents[1] / "jang_tools" / "turboquant" / "gather_tq_kernel.py").read_text()

    assert "def make_gather_tq_decode_broadcast" in src
    assert "K_meta = K" in src


def test_gather_kernel_exposes_weighted_decode_helper_for_moe_down_projection():
    src = (Path(__file__).resolve().parents[1] / "jang_tools" / "turboquant" / "gather_tq_kernel.py").read_text()

    assert "def make_weighted_gather_tq_decode_per_row" in src
    assert "weighted_gather_tq_matmul" in src
    assert 'input_names=["x_rot", "packed", "norms", "codebook", "rhs_indices", "scores", "meta"]' in src
    assert "output_shapes=[(1, out_features)]" in src


def test_switchglu_decode_installs_weighted_decode_fast_path():
    src = (Path(__file__).resolve().parents[1] / "jang_tools" / "jangrt" / "switchglu_decode.py").read_text()

    assert "_jangtq_weighted_decode" in src
    assert "scores_flat" in src
    assert "make_weighted_gather_tq_decode_per_row" in src
    assert "JANGTQ_ENABLE_WEIGHTED_DOWN_GATHER" in src
    assert "def _weighted_sum_mlp" in src
    assert "mx.compile(_weighted_sum_mlp)" in src
    assert "mx.sum(y * scores_flat.astype(y.dtype)[:, None], axis=0" in src


def test_switchglu_decode_exposes_sort_threshold_for_prefill_tuning():
    src = (Path(__file__).resolve().parents[1] / "jang_tools" / "jangrt" / "switchglu_decode.py").read_text()

    assert "JANGTQ_SWITCHGLU_SORT_THRESHOLD" in src
    assert "_sort_threshold" in src


def test_load_jangtq_patches_nemotron_activation_dtype_widening():
    src = (Path(__file__).resolve().parents[1] / "jang_tools" / "load_jangtq.py").read_text()

    assert "JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16" in src
    assert "_patched_nemotron_block_call" in src
    assert "_patched_nemotron_model_call" in src
    assert "out.astype(x.dtype)" in src
    assert "out.astype(_target_dtype)" in src


def test_load_jangtq_patches_nemotron_weighted_switch_mlp_decode():
    src = (Path(__file__).resolve().parents[1] / "jang_tools" / "load_jangtq.py").read_text()

    assert "_get_compiled_switchmlp_weighted_decode" in src
    assert "_jangtq_weighted_decode" in src
    assert "_patched_nemotron_moe_call" in src
    assert "JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH" in src
    assert "mx.sum(y * scores_flat.astype(y.dtype)[:, None], axis=0" in src
