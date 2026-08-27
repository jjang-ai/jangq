"""HF side of the qwen4_exp parity test (runs in parity_venv, torch CPU fp32).

Builds a tiny Qwen4ExpForCausalLM, random-inits, saves:
  - weights in REAL-checkpoint naming (model.language_model.*, lm_head.*,
    fused experts layout, ngram table split into 12 shards, conv [C,1,K])
  - input ids
  - reference logits + per-layer hidden states
"""

import numpy as np
import torch

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

torch.manual_seed(11)

cfg = Qwen4ExpTextConfig(
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    vocab_size=997,
    intermediate_size=128,
    full_attention_interval=4,
    linear_num_value_heads=6,
    linear_num_key_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=8,
    num_experts_per_tok=3,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    hc_count=4,
    hc_lowrank=16,
    ple_layer_ids=[2],
    ple_embed_dim=64,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=8,
    ngram_vocab_size_base=1009,
    make_ngram_vocab_size_divisible_by=128,
    seed=1234,
    split_ngram_parts=12,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=32,
    indexer_budget=8,
    indexer_compress_ratio=4,
    output_gate_type="sigmoid",
    rms_norm_eps=1e-6,
    eos_token_id=7,
    bos_token_id=1,
    pad_token_id=None,
    tie_word_embeddings=False,
    max_position_embeddings=4096,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
    },
    attn_implementation="eager",
)

model = Qwen4ExpForCausalLM(cfg).eval().float()

# randomize everything with a healthy scale (default init leaves norms at 0→1)
sd = model.state_dict()
gen = torch.Generator().manual_seed(23)
for k, v in sd.items():
    if v.dtype in (torch.int64, torch.int32, torch.long):
        continue
    if "layer_multipliers" in k or "ngram_heads" in k:
        continue
    sd[k] = torch.randn(v.shape, generator=gen) * 0.5 / max(1, v.shape[-1]) ** 0.5
model.load_state_dict(sd)

rng = np.random.default_rng(3)
S = 29
ids = rng.integers(0, cfg.vocab_size, size=(1, S))
ids[0, 11] = cfg.eos_token_id
input_ids = torch.tensor(ids, dtype=torch.long)

with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)

logits = out.logits.numpy()
hiddens = {f"__hidden_{i}__": h.numpy() for i, h in enumerate(out.hidden_states)}

# ---- export weights in real-checkpoint format ----
export = {}
for k, v in model.state_dict().items():
    if v.dtype in (torch.long, torch.int32):
        v = v.to(torch.int64)
        arr = v.numpy()
    else:
        arr = v.float().numpy()
    if k.startswith("model."):
        nk = "model.language_model." + k[len("model."):]
    else:
        nk = k  # lm_head.weight
    if nk.endswith("ple.ple_embedding.ngram_embedding.weight"):
        # split into 12 shards, uneven-last, to exercise numeric-order concat
        n = 12
        rows = arr.shape[0]
        per = int(np.ceil(rows / n))
        for i in range(n):
            part = arr[i * per: (i + 1) * per]
            if part.size:
                export[nk.replace(".weight", f".shard_{i}.weight")] = part
        continue
    export[nk] = arr

np.savez_compressed(
    "/private/tmp/claude-501/-Users-eric-jang/de7c8cec-025f-4630-8bbf-5db4e151da1a/scratchpad/parity_ref.npz",
    __logits__=logits,
    __input_ids__=ids,
    **hiddens,
    **export,
)
print("saved. logits shape", logits.shape, "n_hiddens", len(hiddens),
      "shapes", sorted({h.shape for h in hiddens.values()}),
      "n_weights", len(export))
