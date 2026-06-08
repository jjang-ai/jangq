# Nemotron Ultra Runtime Shape Contract

bundle: `/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`
log_dir: `docs/runtime/logs`
status: `READY`

## Architecture
- family: `nemotron_h`
- modality: `text`
- cache_type: `hybrid`
- hidden_size: `8192`
- vocab_size: `131072`
- tie_word_embeddings: `False`
- num_hidden_layers: `108`
- layer_counts: `{"attention": 12, "mamba": 48, "moe": 48, "total": 108}`
- attention_heads: `64`
- key_value_heads: `2`
- num_experts_per_tok: `22`
- moe_intermediate_size: `5120`
- ssm_state_size: `128`

## Quantization
- mxtq_bits: `{"mamba_projection": 8, "routed_expert": {"down_proj": 1, "up_proj": 1}, "shared_expert": 8}`
- method: `routed-tq1-control-plane-preserved`
- drops_mtp: `True`
- estimated_output_gib: `98.35`
- fp8_projection_affine_bits: `8`
- fp8_projection_group_size: `128`
- keeps_attention_bf16: `True`
- keeps_latent_moe_bf16: `True`
- keeps_router_gates_source_precision: `True`
- shard_count: `51`

## Mamba Decode Contract
- layer_index: `0`
- cache_ordinal: `0`
- hidden_shape: `[1, 1, 8192]`
- normed_shape: `[1, 1, 8192]`
- projected_shape: `[1, 1, 35072]`
- gate_shape: `[1, 1, 16384]`
- conv_dim: `18432`
- conv_input_shape: `[1, 1, 18432]`
- conv_output_shape: `[1, 1, 18432]`
- ssm_out_shape: `[1, 1, 16384]`
- intermediate_size: `16384`
- num_heads: `256`
- n_groups: `8`
- ssm_state_size: `128`

## MoE Decode Contract
- layer_index: `1`
- hidden_shape: `[1, 1, 8192]`
- scores_shape: `[1, 1, 22]`
- indices_shape: `[1, 1, 22]`
- latent_shape: `[1, 1, 2048]`
- routed_shape: `[1, 1, 22, 2048]`

## Preserve
- 48 Mamba companion cache entries plus 12 attention KV entries for hybrid prefix cache
- MTP remains dropped; speculative draft KV/SSM state is out of scope for this bundle
- routed expert up/down projections remain 1-bit according to mxtq_bits
- shared expert and Mamba projection paths remain 8-bit unless new proof reverses the projection tradeoff
- router gates and attention BF16 retention remain source-precision/preserved runtime surfaces
- text-only modality remains explicit; media requests must reject or reroute
