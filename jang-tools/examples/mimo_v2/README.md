# MiMo runtime examples

These scripts inspect or exercise a local model bundle. Use a separate output directory for conversion and retain the original source weights.

- `affine_codec_compare.py`
- `awq_source_probe.py`
- `bundle_audit.py`
- `bundle_config_audit.py`
- `compare_loaded_attention_component.py`
- `compare_loaded_expert_qdq.py`
- `compare_loaded_moe_component.py`
- `decode_component_probe.py`
- `direct_logits_probe.py`
- `estimate_affine_profiles.py`
- `estimate_pruned_affine_profiles.py`
- `expert_quant_probe.py`
- `inspect_loaded_quant.py`
- `layer_component_probe.py`
- `layer_diff_probe.py`
- `moe_prefill_component_probe.py`
- `router_trace_probe.py`
- `source_greedy_probe.py`
- `source_profile_probe.py`
- `source_prune_vs_affine_probe.py`
- `text_smoke.py`
- `vl_smoke.py`
- `vlm_vision_parity_probe.py`

Run a script with `--help` to inspect its arguments. A metadata check or short probe does not establish full runtime or model-quality compatibility.
