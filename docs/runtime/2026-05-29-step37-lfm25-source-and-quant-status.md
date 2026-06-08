# Step 3.7 Flash and LFM2.5 8B-A1B Quant Status

Date: 2026-05-29

## Source Checkpoints

### LFM2.5-8B-A1B

- Hub id: `LiquidAI/LFM2.5-8B-A1B`
- Resolved sha: `e20b8981cea25d2758f541d2cdadccf4906334bd`
- Local source: `/Volumes/EricsLLMDrive/jangq-ai/sources/LFM2.5-8B-A1B`
- Local size: `16G`
- Source tensors: 2,302 in `model.safetensors`
- Config: `model_type=lfm2_moe`, `architectures=["Lfm2MoeForCausalLM"]`
- Shape facts:
  - `hidden_size=2048`
  - `num_hidden_layers=24`
  - `num_experts=32`
  - `num_experts_per_tok=4`
  - `num_dense_layers=2`
  - Layer mix: 18 LIV convolution layers, 6 full-attention layers
  - Attention: 32 query heads, 8 KV heads
  - Conv cache: `conv_L_cache=3`
  - Context: config `128000`, README states `131072`
- Template/runtime facts:
  - ChatML-like template in `chat_template.jinja`
  - Reasoning is explicit `<think>...</think>` in assistant output
  - Tool calls use Liquid Python-call syntax between `<|tool_call_start|>` and `<|tool_call_end|>`
  - Runtime capability stamp should be `family=lfm2_moe`, `reasoning_parser=qwen3`, `tool_parser=lfm2`, `cache_type=hybrid`

### Step-3.7-Flash-NVFP4

- Hub id: `stepfun-ai/Step-3.7-Flash-NVFP4`
- Resolved sha: `f4aeff5ce8459ddedd3d142bba9f1748aecd14d9`
- Local source: `/Volumes/EricsLLMDrive/jangq-ai/sources/Step-3.7-Flash-NVFP4`
- Local size: `116G`
- Shards: 13 safetensors
- Index: 1,888 tensors, 13 referenced shards, no missing shard references
- Config: top-level `model_type=step3p7`, nested text `model_type=step3p5`
- Shape facts from real shards:
  - Embedding: `model.language_model.embed_tokens.weight BF16 [128896, 4096]`
  - Text layers: 45
  - Early dense MLP layers are BF16
  - Routed MoE layers contain packed ModelOpt NVFP4-style tensors:
    - `*.moe.*.weight U8`
    - `*.moe.*.weight_scale F8_E4M3`
    - `*.moe.*.weight_scale_2 F32`
    - `*.moe.*.input_scale F32`
  - Attention and shared experts seen in sampled shards are BF16
- Template/runtime facts:
  - Step template uses XML tool calls:
    `<tool_call><function=...><parameter=...>...</parameter></function></tool_call>`
  - vLLM/SGLang recipes use `reasoning-parser step3p5` and `tool-call-parser step3p5`
  - Local vMLX/JANG capability should route Step through `reasoning_parser=qwen3`, `tool_parser=step3p5`, `think_in_template=true`, `cache_type=kv`, `modality=vision`
  - The assistant prefill opens `<think>` in the template; runtimes must not add a second synthetic reasoning prefix.
- JANG_2L dry-run facts from `jang_tools.step37.convert_jang --dry-run`:
  - `nvfp4_payloads=126`, `nvfp4_sidecars=378`, `bf16_quantized=404`, `bf16_passthrough=980`
  - `vision_tensors=667`, `audio_tensors=0`, `mtp_tensors=0`
  - Bit allocation after protecting head-wise attention gate:
    - `self_attn.{q,k,v,o,g}_proj`: 8-bit affine
    - `embed_tokens`: 6-bit affine
    - routed experts: `gate_proj=4`, `down_proj=3`, `up_proj=2`
    - vision encoder/projector: BF16 passthrough for the first runtime bundle
  - Group sizing: 288 routed experts selects MLX `group_size=128` for expert/attention fast path; true MoE router gates stay `group_size=64` if present.
  - Estimated output payload: about `81.604 GiB` before filesystem overhead.
  - Size by component:
    - routed MoE: `72.031 GiB`
    - attention: `4.342 GiB`
    - vision/projector: `3.690 GiB`
    - embeddings/lm_head: `0.891 GiB`
    - dense/shared MLP: `0.649 GiB`
- Runtime implementation boundary:
  - The downloaded checkpoint is VLM-only: vision/projector are present; no audio tensors or audio tokenizer files were found.
  - The source config advertises next-token prediction layers, but no MTP/nextn tensors are present in the NVFP4 source. Do not create an MTP bundle from config alone.
  - A coherent local generation proof still requires a Step3p7/Step3p5 runtime path in vMLX/MLX that implements full+sliding KV cache, head-wise attention gate, q/k norms, K/V scale sidecars, and image patch handling. The quantized bundle alone is not runtime proof.

### Step-3.7-Flash Full BF16

- Hub id: `stepfun-ai/Step-3.7-Flash`
- Resolved sha: `5371d3c294b4f7f8917538ed64f039c060432eda`
- Size from Hub manifest: `375.082 GiB`
- Current blocker: does not fit on `/Volumes/EricsLLMDrive` after current sources; free space was about `139Gi` after the NVFP4 download.
- Do not claim BF16 Step conversion has started unless this checkpoint is actually present or enough space is reclaimed.

## Built LFM Artifacts

### MXFP4

- Output: `/Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-MXFP4`
- Size: `4.2G`
- Command:

```sh
/Users/eric/jang/jang-tools/.venv/bin/python -m mlx_lm convert \
  --hf-path /Volumes/EricsLLMDrive/jangq-ai/sources/LFM2.5-8B-A1B \
  --mlx-path /Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-MXFP4 \
  -q --q-mode mxfp4 --q-group-size 32
```

- Converter output: `Quantized model with 4.251 bits per weight.`
- Verified layout:
  - `model.embed_tokens.weight U32 [128000, 256]`
  - `model.embed_tokens.scales U8 [128000, 64]`
  - LFM depthwise conv state stays BF16 as `model.layers.0.conv.conv.weight BF16 [2048, 3, 1]`
- Smoke:
  - Prompt: `What is 2+2? Answer briefly.`
  - Generated reasoning identified `4`
  - Reported speed in `mlx_lm.generate`: about `286 tok/s` on a 96-token run, peak memory about `4.544 GB`

### MXFP8

- Output: `/Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-MXFP8`
- Size: `8.1G`
- Command:

```sh
/Users/eric/jang/jang-tools/.venv/bin/python -m mlx_lm convert \
  --hf-path /Volumes/EricsLLMDrive/jangq-ai/sources/LFM2.5-8B-A1B \
  --mlx-path /Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-MXFP8 \
  -q --q-mode mxfp8 --q-group-size 32
```

- Converter output: `Quantized model with 8.250 bits per weight.`
- Verified layout:
  - `model.embed_tokens.weight U32 [128000, 512]`
  - `model.embed_tokens.scales U8 [128000, 64]`
- Smoke:
  - Prompt: `What is 2+2? Answer briefly.`
  - Generated reasoning identified `4`
  - Reported speed in `mlx_lm.generate`: about `196 tok/s` on a 64-token run, peak memory about `8.767 GB`

### JANG_2L

- Output: `/Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-JANG_2L`
- Size: `2.9G`
- JANG runtime size metadata: `2.84 GB`
- Command:

```sh
/Users/eric/jang/jang-tools/.venv/bin/python -m jang_tools --progress json convert \
  /Volumes/EricsLLMDrive/jangq-ai/sources/LFM2.5-8B-A1B \
  -o /Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-JANG_2L \
  -p JANG_2L
```

- Final allocation:
  - Actual bits: `2.37`
  - Block size: `64`
  - Bit widths used: `2`, `6`, `8`
  - Passthrough bit widths: `16`
- Required converter fixes:
  - Add `lfm2_moe` capability stamp: `qwen3` reasoning, `lfm2` tools, `hybrid` cache
  - Treat LFM as hybrid conv + attention + MoE, not plain MoE
  - Preserve and transpose `conv.conv.weight` to MLX Conv1d layout `[2048, 3, 1]`
  - Map dense source `feed_forward.w1/w2/w3` to MLX `gate_proj/down_proj/up_proj`
  - Protect `self_attn.out_proj`, LFM conv projections, and always-active dense `feed_forward.w1/w2/w3` above the generic 2-bit expert floor
- Verified JANG keys:
  - `model.layers.0.feed_forward.gate_proj.*`
  - `model.layers.0.feed_forward.down_proj.*`
  - `model.layers.0.feed_forward.up_proj.*`
  - `model.layers.0.conv.conv.weight F16 [2048, 3, 1]`
- Smoke:

```sh
/Users/eric/jang/jang-tools/.venv/bin/python -m jang_tools inference \
  --model /Users/eric/.mlxstudio/models/JANGQ-AI/LFM2.5-8B-A1B-JANG_2L \
  --prompt "What is 2+2? Answer briefly." \
  --max-tokens 128 --temperature 0 --json
```

- Result:
  - Output closed `<think>...</think>` and answered `2 + 2 = 4.`
  - Reported speed: `206.886 tok/s`
  - Load time: `1.946 s`
  - Peak RSS: `3887 MB`

## Step JANG_2L Status

Step JANG_2L is structurally built locally.

- Output: `/Users/eric/.mlxstudio/models/JANGQ-AI/Step-3.7-Flash-JANG_2L`
- Size: `82G`
- Shards: 67
- Indexed tensors: 2,570
- Index total size: `87,621,944,984` bytes
- Capability verification: `capabilities OK (family=step3p7)`
- Raw NVFP4 sidecars in output index: none (`weight_scale=0`, `weight_scale_2=0`, `input_scale=0`)
- Critical layout checks:
  - `model.language_model.layers.0.self_attn.g_proj.weight U32 [64, 1024]`
  - `model.language_model.layers.3.moe.gate_proj.weight U32 [288, 1280, 512]`
  - `model.language_model.layers.3.moe.down_proj.weight U32 [288, 4096, 120]`
  - `model.language_model.layers.3.moe.up_proj.weight U32 [288, 1280, 256]`
  - `model.vision_model.conv1.weight F16 [1536, 3, 14, 14]`
  - `model.vit_large_projector.weight F16 [4096, 6144]`

The converter hit and fixed one real source-layout issue during the first run: some ModelOpt NVFP4 scale sidecars live in different safetensors shards from their `*.weight` payloads. The writer now resolves sidecars through `model.safetensors.index.json` instead of assuming same-shard locality.

Text coherence proof now passes through the local `step3p7_mlx.py` bridge over `mlx_lm.models.step3p5`.

- Prompt: `What is 2+2? Answer with only the number.`
- Output ended with `</think>\n4`
- Prompt tokens: 26
- Generated tokens: 58
- Fresh post-upload coherence proof:
  - Prefill: `9.084182977676392 s`
  - Total: `16.710078239440918 s`
  - Decode speed after prefill: `7.605664385505815 tok/s`

That short coherence proof is a cold/short decode measurement and should not be used as the steady-state speed number. A no-wrapper warmed decode run verified the MLX quantized path and measured:

- `mx.quantized_matmul` calls during proof: `7254`
- `mx.gather_qmm` calls during proof: `2268`
- `mx.matmul` calls during proof: `0`
- Warmed measured tokens: 32
- Warmed decode time: `0.7534263134002686 s`
- Warmed decode speed: `42.47263392697507 tok/s`

Representative loaded module checks:

- `self_attn.q_proj`: `mlx.nn.layers.quantized.QuantizedLinear`, 8-bit, group size 128
- `self_attn.g_proj`: `mlx.nn.layers.quantized.QuantizedLinear`, 8-bit, group size 128
- routed `switch_mlp.gate_proj`: `QuantizedSwitchLinear`, 4-bit, group size 128
- routed `switch_mlp.up_proj`: `QuantizedSwitchLinear`, 2-bit, group size 128
- routed `switch_mlp.down_proj`: `QuantizedSwitchLinear`, 3-bit, group size 128
- router gate: `QuantizedLinear`, 8-bit, group size 64

Full VLM coherence is not proven yet. The local artifact is text-coherent and structurally verified, but image input still requires Step3p7 VLM wrapper/image patch runtime proof.

Tokenizer metadata fix: `tokenizer_class=PreTrainedTokenizerFast` is required. The source metadata causes HF auto-load to choose a Llama tokenizer class and display byte-level markers (`Ġ`, `Ċ`) in decoded text even though `tokenizer.json` itself decodes correctly.

## Step JANGTQ-2K Prep

Step's routed MoE tensors are balanced by projection:

- `moe.gate_proj`: `63,417,876,480` logical weights
- `moe.up_proj`: `63,417,876,480` logical weights
- `moe.down_proj`: `63,417,876,480` logical weights
- total routed expert weights: `190,253,629,440`

Current Step JANG_2L routed affine storage is about `77.291 GB`:

- `gate_proj` 4-bit affine: `33.691 GB`
- `up_proj` 2-bit affine: `17.836 GB`
- `down_proj` 3-bit affine: `25.764 GB`

Candidate JANGTQ mixed projection profiles, assuming pre-stacked `.tq_packed/.tq_norms/.tq_bits` routed tensors and unchanged non-routed tensors:

| Profile (`gate/up/down`) | Routed TQ size | Total bundle estimate | Notes |
| --- | ---: | ---: | --- |
| `2/2/2` | `47.724 GB` | `~58.055 GB` | smallest, highest quality risk |
| `2/4/2` | `63.579 GB` | `~73.910 GB` | protects `up_proj`; same size as any one-4-bit profile |
| `4/2/2` | `63.579 GB` | `~73.910 GB` | protects gating branch; best first 2K candidate |
| `2/2/4` | `63.579 GB` | `~73.910 GB` | protects output/down projection; chosen for `JANGTQ_K` because gate/up share one fused-kernel bit width |
| `4/2/3` | `71.506 GB` | `~81.837 GB` | closest to current JANG_2L projection policy |
| `4/4/2` | `79.433 GB` | `~89.764 GB` | larger than current routed affine; not a size win |

Initial Step `JANGTQ_2K` candidate was `4/2/2` for `gate_proj/up_proj/down_proj`.

Updated result: `JANGTQ_K` uses `2/2/4`. This keeps `gate_proj` and `up_proj` at the same bit width, so the existing fused gate+up TQ kernel remains valid, and spends the 4-bit budget on `down_proj`.

JANGTQ bundle requirements:

- emit routed experts as pre-stacked `model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.tq_packed/.tq_norms/.tq_bits`
- keep top-level `mxtq_seed=42`
- keep dual-form metadata: top-level `mxtq_bits` and `jang_config.quantization.mxtq_bits`
- include `routed_expert_bits` matching the shipped profile
- build and ship `jangtq_runtime.safetensors`
- verify text runtime with TQ hydration and no missing required TQ swaps before upload

### Step 4/2/2 Build Results

`JANGTQ_2K` was built locally on `/Volumes/EricsLLMDrive` and then rejected after runtime proof:

- profile: `JANGTQ_2K`
- routed bits: `gate_proj=4`, `up_proj=2`, `down_proj=2`
- size: about `69G`
- shards: 55
- indexed tensors: 2,570
- TQ triplets: 126 `tq_packed`, 126 `tq_norms`, 126 `tq_bits`
- sidecar: `jangtq_runtime.safetensors` present
- TQ sidecar entries:
  - `signs.1280.42`
  - `signs.4096.42`
  - `codebook.1280.2`
  - `codebook.4096.2`
  - `codebook.4096.4`

Two concrete runtime issues were found and fixed/documented:

- Per-module `config.quantization` entries must not contain `format="mxtq"`. MLX passes those dictionaries directly to `SwitchLinear.to_quantized()`, which does not accept `format`.
- Pre-stacked Step TQ keys must be emitted under `model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}` for the `step3p7_mlx.py` text bridge. Raw `model.language_model.layers...` TQ keys hydrate zero modules.

After those fixes, Python JANGTQ hydration worked mechanically:

- `TQ groups: 126`
- `pre-stacked: 126`
- `Replaced 126 modules`
- `P18 QKV fusion: 45 instances`
- decode speed on the failed smoke: about `36-38 tok/s`

The later root cause was not TQ stacking. The incoherent `Huawei ...` loops were caused by the generic P18 QKV fusion in `load_jangtq.py` being incorrect for Step3p5 attention:

- q/k RMSNorm must be applied after reshaping to `(B, L, heads, head_dim)`, not on the flat projection.
- Step head-wise attention gate must be preserved: `attn_out *= sigmoid(g_proj(x))[..., None]`.

After fixing P18, the affine `JANG_2L` proof recovered, and a stacked `JANGTQ_4K` control also answered the Step chat-template arithmetic smoke correctly. The final target is `JANGTQ_K`, not the rejected `JANGTQ_2K`.

Runtime contract for future vMLX Python/Swift work:

- Mixed `gate_proj`/`up_proj` bits are not compatible with the current fused gate+up TQ kernels because the fused kernels take one `bits` value for both gate and up packed tensors.
- For `gate=4, up=2`, runtimes need either:
  - a true mixed-bit fused gate+up kernel that accepts separate gate/up bit widths, packed tensors, and codebooks, or
  - a correct unfused fallback: `up_proj(x, idx)`, `gate_proj(x, idx)`, activation, then `down_proj(...)`.
- Swift/vMLX should fail fast if TQ keys hydrate zero modules or if mixed gate/up bits are routed through a single-bit fused gate+up kernel.
- P18-style QKV fusion must be architecture-aware for Step:
  - q/k RMSNorm is per-head, after reshape.
  - head-wise attention gate is required.
  - Dropping either produces repetitive gibberish even when quantization and cache are otherwise correct.

`JANGTQ_K` was built and verified after the P18 fix:

- output: `/Volumes/EricsLLMDrive/jangq-ai/Step-3.7-Flash-JANGTQ_K`
- profile: `JANGTQ_K`
- routed bits: `gate_proj=2`, `up_proj=2`, `down_proj=4`
- size: about `69G`
- index total size: `73,910,302,350` bytes
- shards: 55
- indexed tensors: 2,570
- TQ triplets: 126 `tq_packed`, 126 `tq_norms`, 126 `tq_bits`
- raw NVFP4 sidecars: 0
- bad `model.language_model.*.tq_*` keys: 0
- hydration proof: `TQ groups: 126`, `pre-stacked: 126`, `Replaced 126 modules`
- live routed module bits: 42 switch layers with `(gate, up, down) = (2, 2, 4)`
- text proof: Step chat-template prompt `What is 2+2? Answer with only the number.` ended with `</think>\n4`
- warm smoke: `64` generated-token cap completed in `1.790820837020874 s` on the local M5 Max proof run

`JANG_K` was built and verified as the affine comparison point:

- output: `/Volumes/EricsLLMDrive/jangq-ai/Step-3.7-Flash-JANG_K`
- profile: `JANG_K`
- routed bits: `gate_proj=4`, `up_proj=2`, `down_proj=2`
- size: about `74G`
- index total size: `79,694,709,712` bytes
- shards: 58
- indexed tensors: 2,570
- no TQ tensors
- no raw NVFP4 sidecars
- capability verification: `capabilities OK (family=step3p7)`
- text proof output ended with `</think>\n4`
- warmed decode: `32` tokens in `0.8008251190185547 s` = `39.95878655656726 tok/s`

## Hugging Face Uploads

Uploaded and read back from the Hub:

- `OsaurusAI/LFM2.5-8B-A1B-JANG_2L`
  - Public: yes
  - Commit: `1996517a5ca2e8f74947685cfc0b132cbaecadfd`
  - Files: README, LICENSE, chat template, config, generation config, tokenizer, JANG config, 6 safetensor shards, index
- `OsaurusAI/LFM2.5-8B-A1B-MXFP4`
  - Public: yes
  - Commit: `97a907274ee1b8e65c066af1e69f79b4ef8104b4`
  - Files: README, LICENSE, chat template, config, generation config, tokenizer, safetensor, index
- `OsaurusAI/LFM2.5-8B-A1B-MXFP8`
  - Public: yes
  - Commit: `dd95e07a4b058443c8d893cb9cc0c763d1864c3d`
  - Files: README, LICENSE, chat template, config, generation config, tokenizer, 2 safetensor shards, index
- `OsaurusAI/Step-3.7-Flash-JANG_2L`
  - Public: yes
  - Commit: `325c3364e0a25de8ffc9890622b99877d1854b14`
  - Files: 83 total including README, chat template, config, generation config, tokenizer, JANG config, MLX text bridge, 67 safetensor shards, index
  - Negative remote check: no `.cache`, `__pycache__`, or `*.pyc` files in Hub siblings
- `JANGQ-AI/LFM2.5-8B-A1B-JANG_2L`
  - Public: yes
  - Commit: `89d765ac337c638c13ba7316f8fe5138279d3eaf`
  - Files: 16 total including README, LICENSE, chat template, config, generation config, tokenizer, JANG config, 6 safetensor shards, index
  - Negative remote check: no `.cache`, `__pycache__`, or `*.pyc` files in Hub siblings
- `JANGQ-AI/Step-3.7-Flash-JANG_2L`
  - Public: yes
  - Commit: `f0de50760bb0af979a40cbef530caba57a211b6f`
  - Files: 83 total including README, chat template, config, generation config, tokenizer, JANG config, MLX text bridge, 67 safetensor shards, index
  - Negative remote check: no `.cache`, `__pycache__`, or `*.pyc` files in Hub siblings
- `JANGQ-AI/Step-3.7-Flash-JANG_K`
  - Public: yes
  - Commit: `37375de14375f32965f4056a96edf45196593869`
  - Files: 74 total including README, chat template, config, generation config, tokenizer, JANG config, MLX text bridge, 58 safetensor shards, index
  - Negative remote check: no `.cache`, `__pycache__`, or `*.pyc` files in Hub siblings
- `OsaurusAI/Step-3.7-Flash-JANG_K`
  - Public: yes
  - Commit: `97f8294c21c3aa825f5b2764dd2cf692f9f07d21`
  - Files: 74 total including README, chat template, config, generation config, tokenizer, JANG config, MLX text bridge, 58 safetensor shards, index
  - Negative remote check: no `.cache`, `__pycache__`, or `*.pyc` files in Hub siblings

## Runtime Notes

- LFM cache type is hybrid: attention layers use KV cache, LIV conv layers use array/state cache.
- LFM reasoning parser should not assume the template pre-opens `<think>`; `think_in_template=false` is intentional.
- LFM tool parser must handle Liquid Python-call lists inside `<|tool_call_start|>` and `<|tool_call_end|>`.
- For quick quality gates, use at least one prompt that requires the model to close the think block and produce visible final content. A load-only check is insufficient.
- `mlx_lm convert --q-mode ...` is not enough; it must include `-q` or it produces an unquantized MLX-format copy.
