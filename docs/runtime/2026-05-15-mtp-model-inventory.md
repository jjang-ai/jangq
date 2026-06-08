# MTP Model Inventory - 2026-05-15

Generated: `2026-05-15 13:04:16 PDT`
Root scanned: `/Users/eric/models`

This is a static metadata/header inventory. It does not load model weights into MLX and does not prove runtime support. It checks `config.json`, `jang_config.json`, `generation_config.json`, `tokenizer_config.json`, `model.safetensors.index.json`, and safetensors headers when no index exists.

## Summary

| Status | Count | Meaning |
|---|---:|---|
| `present_in_weights` | 1 | MTP tensors present in weights/index |
| `config_claims_mtp_but_no_tensors` | 5 | Config declares MTP but no MTP tensors were found |
| `absent_by_config` | 8 | No MTP by config |
| `unknown_no_mtp_evidence` | 17 | Unknown/no explicit MTP evidence |

## Family counts

| model_type | total | statuses |
|---|---:|---|
| `bailing_hybrid` | 3 | `config_claims_mtp_but_no_tensors`=3 |
| `deepseek_v4` | 4 | `absent_by_config`=3, `present_in_weights`=1 |
| `gemma4` | 1 | `unknown_no_mtp_evidence`=1 |
| `hy_v3` | 1 | `config_claims_mtp_but_no_tensors`=1 |
| `kimi_k25` | 2 | `unknown_no_mtp_evidence`=2 |
| `laguna` | 1 | `unknown_no_mtp_evidence`=1 |
| `minimax_m2` | 6 | `absent_by_config`=5, `config_claims_mtp_but_no_tensors`=1 |
| `nemotron_h` | 3 | `unknown_no_mtp_evidence`=3 |
| `qwen3_5` | 2 | `unknown_no_mtp_evidence`=2 |
| `qwen3_5_moe` | 2 | `unknown_no_mtp_evidence`=2 |
| `unknown` | 1 | `unknown_no_mtp_evidence`=1 |
| `zaya` | 2 | `unknown_no_mtp_evidence`=2 |
| `zaya1_vl` | 3 | `unknown_no_mtp_evidence`=3 |

## Activation rules

- Do not enable MTP from config alone. Require actual MTP tensor keys in the index/header or a reconversion that preserves them.
- Do not mix MTP draft KV/state with the accepted verifier cache. MTP needs a separate draft-cache pool and only accepted tokens may update the normal cache.
- `present_in_weights` / `present_disabled` means runtime work can start from this artifact.
- `dropped_by_jang` / `config_claims_mtp_but_no_tensors` means reconvert or restore the MTP tensors first; a scheduler flag cannot recover missing weights.
- `absent_by_config` should stay disabled and becomes negative-space proof for no behavior regression.

## Highest-value targets

- `~/models/Sources/DeepSeek-V4-Flash` — `present_in_weights`, model_type=`deepseek_v4`, mtp_tensor_keys=1575. Evidence: num_nextn_predict_layers=1. Next: Use DSV4-specific MTP/compressor/indexer path; requires MTP-preserved artifact and DSV4 composite-cache verifier branch.
- `~/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK` — `config_claims_mtp_but_no_tensors`, model_type=`bailing_hybrid`, mtp_tensor_keys=0. Evidence: num_nextn_predict_layers=1; mxtq_bits.mtp_eh_proj=8. Next: Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights.
- `~/models/dealign.ai/Ling-2.6-flash-MXFP4-CRACK` — `config_claims_mtp_but_no_tensors`, model_type=`bailing_hybrid`, mtp_tensor_keys=0. Evidence: num_nextn_predict_layers=1. Next: Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights.
- `~/models/dealign.ai/MiniMax-M2.7-JANGTQ-CRACK` — `config_claims_mtp_but_no_tensors`, model_type=`minimax_m2`, mtp_tensor_keys=0. Evidence: num_mtp_modules=3; mtp_transformer_layers=1; use_mtp=True. Next: Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights.
- `~/models/JANGQ/Hy3-preview-JANGTQ2` — `config_claims_mtp_but_no_tensors`, model_type=`hy_v3`, mtp_tensor_keys=0. Evidence: num_nextn_predict_layers=1; mxtq_bits.mtp=8; runtime.bundle_has_mtp=True; runtime.mtp_layers=1; runtime.mtp_mode='preserved_disabled'; runtime.mtp_status='MTP tensors are preserved in the bundle, but the first JANG runtime path must use normal autoregressive decode until an accept/reject speculative loop is implemented and tested.'. Next: Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights.
- `~/models/JANGQ/Ling-2.6-flash-JANGTQ` — `config_claims_mtp_but_no_tensors`, model_type=`bailing_hybrid`, mtp_tensor_keys=0. Evidence: num_nextn_predict_layers=1; mxtq_bits.mtp_eh_proj=8. Next: Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights.

## MTP tensors present in weights/index

| Model | model_type | arch | MTP fields | tensor keys | layer range | Activation note |
|---|---|---|---:|---:|---|---|
| `~/models/Sources/DeepSeek-V4-Flash` | `deepseek_v4` | `DeepseekV4ForCausalLM` | 1 | 1575 | `0..42 (43)` | Use DSV4-specific MTP/compressor/indexer path; requires MTP-preserved artifact and DSV4 composite-cache verifier branch. |

### Details

#### `~/models/Sources/DeepSeek-V4-Flash`

- status: `present_in_weights`
- model_type: `deepseek_v4`
- config MTP evidence: `num_nextn_predict_layers=1`
- MTP tensor keys found: `1575`
- key: `mtp.0.hc_head_base`
- key: `mtp.0.hc_head_fn`
- key: `mtp.0.hc_head_scale`
- key: `mtp.0.hc_attn_base`
- key: `mtp.0.hc_ffn_base`
- key: `mtp.0.hc_attn_fn`
- key: `mtp.0.hc_attn_scale`
- key: `mtp.0.hc_ffn_fn`
- key: `mtp.0.hc_ffn_scale`
- key: `mtp.0.attn.attn_sink`
- key: `mtp.0.attn.wq_a.weight`
- key: `mtp.0.attn.wq_a.scale`
- key: `mtp.0.attn.wq_b.weight`
- key: `mtp.0.attn.wq_b.scale`
- key: `mtp.0.attn.q_norm.weight`
- key: `mtp.0.attn.wo_a.weight`
- key: `mtp.0.attn.wo_a.scale`
- key: `mtp.0.attn.wkv.weight`
- key: `mtp.0.attn.wkv.scale`
- key: `mtp.0.attn.kv_norm.weight`
- key: `mtp.0.attn.wo_b.weight`
- key: `mtp.0.attn.wo_b.scale`
- key: `mtp.0.attn_norm.weight`
- key: `mtp.0.ffn_norm.weight`
- files with MTP keys: `model-00046-of-00046.safetensors`
- next: Use DSV4-specific MTP/compressor/indexer path; requires MTP-preserved artifact and DSV4 composite-cache verifier branch.

## Config declares MTP but no MTP tensors were found

| Model | model_type | arch | MTP fields | tensor keys | layer range | Activation note |
|---|---|---|---:|---:|---|---|
| `~/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK` | `bailing_hybrid` | `BailingMoeV2_5ForCausalLM` | 5 | 0 | `0..32 (33)` | Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights. |
| `~/models/dealign.ai/Ling-2.6-flash-MXFP4-CRACK` | `bailing_hybrid` | `BailingMoeV2_5ForCausalLM` | 2 | 0 | `0..32 (33)` | Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights. |
| `~/models/dealign.ai/MiniMax-M2.7-JANGTQ-CRACK` | `minimax_m2` | `MiniMaxM2ForCausalLM` | 3 | 0 | `0..61 (62)` | Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights. |
| `~/models/JANGQ/Hy3-preview-JANGTQ2` | `hy_v3` | `HYV3ForCausalLM` | 14 | 0 | `0..80 (81)` | Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights. |
| `~/models/JANGQ/Ling-2.6-flash-JANGTQ` | `bailing_hybrid` | `BailingMoeV2_5ForCausalLM` | 5 | 0 | `0..32 (33)` | Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights. |

### Details

#### `~/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK`

- status: `config_claims_mtp_but_no_tensors`
- model_type: `bailing_hybrid`
- config MTP evidence: `num_nextn_predict_layers=1`
- jang_config MTP evidence: `mxtq_bits.mtp_eh_proj=8`
- MTP tensor keys found: `0`
- next: Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights.

#### `~/models/dealign.ai/Ling-2.6-flash-MXFP4-CRACK`

- status: `config_claims_mtp_but_no_tensors`
- model_type: `bailing_hybrid`
- config MTP evidence: `num_nextn_predict_layers=1`
- MTP tensor keys found: `0`
- next: Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights.

#### `~/models/dealign.ai/MiniMax-M2.7-JANGTQ-CRACK`

- status: `config_claims_mtp_but_no_tensors`
- model_type: `minimax_m2`
- config MTP evidence: `num_mtp_modules=3; mtp_transformer_layers=1; use_mtp=True`
- MTP tensor keys found: `0`
- next: Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights.

#### `~/models/JANGQ/Hy3-preview-JANGTQ2`

- status: `config_claims_mtp_but_no_tensors`
- model_type: `hy_v3`
- config MTP evidence: `num_nextn_predict_layers=1`
- jang_config MTP evidence: `mxtq_bits.mtp=8; runtime.bundle_has_mtp=True; runtime.mtp_layers=1; runtime.mtp_mode='preserved_disabled'; runtime.mtp_status='MTP tensors are preserved in the bundle, but the first JANG runtime path must use normal autoregressive decode until an accept/reject speculative loop is implemented and tested.'; bundle_has_mtp=True; mtp_layers=1`
- MTP tensor keys found: `0`
- next: Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights.

#### `~/models/JANGQ/Ling-2.6-flash-JANGTQ`

- status: `config_claims_mtp_but_no_tensors`
- model_type: `bailing_hybrid`
- config MTP evidence: `num_nextn_predict_layers=1`
- jang_config MTP evidence: `mxtq_bits.mtp_eh_proj=8`
- MTP tensor keys found: `0`
- next: Need reconversion/preservation first; runtime toggle alone cannot activate missing MTP weights.

## No MTP by config

| Model | model_type | arch | MTP fields | tensor keys | layer range | Activation note |
|---|---|---|---:|---:|---|---|
| `~/models/dealign.ai/MiniMax-M2.7-JANGTQ_K-CRACK` | `minimax_m2` | `MiniMaxM2ForCausalLM` | 3 | 0 | `0..61 (62)` | No MTP activation path for this artifact; leave disabled and use as negative-space regression row. |
| `~/models/JANGQ/_dsv4_jangtq_k_upload_prep/stage_jangq` | `deepseek_v4` | `DeepseekV4ForCausalLM` | 2 | 0 | `0..42 (43)` | No MTP activation path for this artifact; leave disabled and use as negative-space regression row. |
| `~/models/JANGQ/_dsv4_jangtq_k_upload_prep/stage_osaurus` | `deepseek_v4` | `DeepseekV4ForCausalLM` | 2 | 0 | `0..42 (43)` | No MTP activation path for this artifact; leave disabled and use as negative-space regression row. |
| `~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K` | `deepseek_v4` | `DeepseekV4ForCausalLM` | 2 | 0 | `0..42 (43)` | No MTP activation path for this artifact; leave disabled and use as negative-space regression row. |
| `~/models/JANGQ/MiniMax-M2.7-JANG_K` | `minimax_m2` | `MiniMaxM2ForCausalLM` | 3 | 0 | `0..61 (62)` | No MTP activation path for this artifact; leave disabled and use as negative-space regression row. |
| `~/models/JANGQ/MiniMax-M2.7-JANGTQ` | `minimax_m2` | `MiniMaxM2ForCausalLM` | 3 | 0 | `0..61 (62)` | No MTP activation path for this artifact; leave disabled and use as negative-space regression row. |
| `~/models/JANGQ/MiniMax-M2.7-JANGTQ_K` | `minimax_m2` | `MiniMaxM2ForCausalLM` | 3 | 0 | `0..61 (62)` | No MTP activation path for this artifact; leave disabled and use as negative-space regression row. |
| `~/models/JANGQ/MiniMax-M2.7-Small-JANGTQ` | `minimax_m2` | `MiniMaxM2ForCausalLM` | 3 | 0 | `0..61 (62)` | No MTP activation path for this artifact; leave disabled and use as negative-space regression row. |

## Unknown/no explicit MTP evidence

| Model | model_type | arch | MTP fields | tensor keys | layer range | Activation note |
|---|---|---|---:|---:|---|---|
| `~/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK` | `gemma4` | `Gemma4ForConditionalGeneration` | 0 | 0 | `0..29 (30)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK` | `nemotron_h` | `NemotronHForCausalLM` | 0 | 0 | `0..51 (52)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ4-CRACK` | `nemotron_h` | `NemotronHForCausalLM` | 0 | 0 | `0..51 (52)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/dealign.ai/Nemotron-Omni-Nano-MXFP4-CRACK` | `nemotron_h` | `NemotronHForCausalLM` | 0 | 0 | `0..51 (52)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 2 | 0 | `0..63 (64)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/dealign.ai/Qwen3.6-27B-MXFP4-CRACK` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 2 | 0 | `0..63 (64)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 2 | 0 | `0..39 (40)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/JANGQ/Kimi-K2.6-Small-JANGTQ` | `kimi_k25` | `KimiK25ForConditionalGeneration` | 1 | 0 | `0..60 (61)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/JANGQ/Laguna-XS.2-JANGTQ` | `laguna` | `LagunaForCausalLM` | 0 | 0 | `0..39 (40)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/JANGQ/ZAYA1-8B-JANGTQ_K` | `zaya` | `ZayaForCausalLM` | 0 | 0 | `0..79 (80)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/JANGQ/ZAYA1-8B-MXFP4` | `zaya` | `ZayaForCausalLM` | 0 | 0 | `0..79 (80)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/JANGQ/ZAYA1-VL-8B-JANGTQ4` | `zaya1_vl` | `Zaya1VLForConditionalGeneration` | 0 | 0 | `0..39 (40)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/JANGQ/ZAYA1-VL-8B-JANGTQ_K` | `zaya1_vl` | `Zaya1VLForConditionalGeneration` | 0 | 0 | `0..39 (40)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/JANGQ/ZAYA1-VL-8B-MXFP4` | `zaya1_vl` | `Zaya1VLForConditionalGeneration` | 0 | 0 | `0..39 (40)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/Kimi-K2.6-JANGTQ_K` | `kimi_k25` | `KimiK25ForConditionalGeneration` | 1 | 0 | `0..60 (61)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/Qwen3.6-35B-A3B-4bit` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 2 | 0 | `0..39 (40)` | Unknown; inspect source/config/index manually before runtime claims. |
| `~/models/Sources/DeepSeek-V4-Flash/inference` | `None` | `None` | 0 | 0 | `` | Unknown; inspect source/config/index manually before runtime claims. |
