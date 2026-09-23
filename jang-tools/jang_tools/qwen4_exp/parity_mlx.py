"""MLX side of the qwen4_exp parity test (runs in jang-tools venv)."""

import numpy as np
import mlx.core as mx

from jang_tools.qwen4_exp.modeling import Model, Qwen4ExpTextArgs
from jang_tools.qwen4_exp.load import sanitize
from jang_tools.qwen4_exp.test_smoke import tiny_args

data = np.load(
    "/private/tmp/claude-501/-Users-eric-jang/de7c8cec-025f-4630-8bbf-5db4e151da1a/scratchpad/parity_ref.npz"
)
weights = {}
hiddens = {}
for k in data.files:
    if k == "__logits__":
        ref_logits = data[k]
    elif k == "__input_ids__":
        ids = data[k]
    elif k.startswith("__hidden_"):
        hiddens[int(k.strip("_").split("_")[1])] = data[k]
    else:
        weights[k] = mx.array(data[k])

args = tiny_args()
model = Model(args)

text, mtp, visual = sanitize(weights, args)

# assert the recomputed n-gram buffers match HF's
ple = model.language_model.layers[1].ple
for k in [k for k in text if k.startswith("__buffer__")]:
    v = np.asarray(text.pop(k)).astype(np.int64)
    name = k.split(".")[-1]
    ref = {
        "layer_multipliers": ple.hasher.layer_multipliers,
        "ngram_heads_offsets": np.array(ple.hasher.head_offsets),
        "ngram_heads_vocab_sizes": np.array(ple.hasher.head_vocab_sizes),
    }[name]
    assert (v == np.asarray(ref)).all(), f"{name} mismatch"
print("ngram buffers match")

model.load_weights(list(text.items()), strict=True)
mx.eval(model.parameters())

ids_mx = mx.array(ids)

for i in sorted(hiddens):
    print(f"hf hidden {i}: {hiddens[i].shape}")

# per-layer capture. HF ordering: hidden_0 = tiled embeddings (wide),
# hidden_{i+1} = layer i output (wide), except the final 64-wide entry which
# is the post-mixer state.
h = model.language_model.embed_tokens(ids_mx)
h = mx.tile(h, (1, 1, args.hc_count))
np.testing.assert_allclose(np.asarray(h), hiddens[0], atol=1e-5)
print("tiled embedding OK")

wide = [i for i in sorted(hiddens) if hiddens[i].shape[-1] == h.shape[-1]]
narrow = [i for i in sorted(hiddens) if hiddens[i].shape[-1] != h.shape[-1]]
layer_refs = wide[1:]  # after the embedding entry

for i, layer in enumerate(model.language_model.layers):
    h = layer(h, mask=None, cache=None, input_ids=ids_mx)
    got = np.asarray(h)
    if i < len(layer_refs):
        want = hiddens[layer_refs[i]]
        d = np.abs(got - want).max()
        rel = d / (np.abs(want).max() + 1e-9)
        status = "OK" if rel < 2e-3 else "MISMATCH"
        print(f"layer {i} ({layer.layer_type}"
              f"{'+PLE' if layer.ple is not None else ''}): max_abs={d:.3e} rel={rel:.3e} {status}")

final = model.language_model.hyper_connection_mixer(h)
if narrow:
    want = hiddens[narrow[-1]]
    d = np.abs(np.asarray(final) - want).max()
    print(f"post-mixer: max_abs={d:.3e}")
logits = np.asarray(model.lm_head(final))
d = np.abs(logits - ref_logits).max()
rel = d / (np.abs(ref_logits).max() + 1e-9)
print(f"logits: max_abs={d:.3e} rel={rel:.3e}")
top_ref = ref_logits.argmax(-1)
top_got = logits.argmax(-1)
agree = (top_ref == top_got).mean()
print(f"top-1 agreement: {agree:.3f}")
assert rel < 5e-3 and agree == 1.0, "PARITY FAILED"
print("FULL-MODEL PARITY vs HF: PASSED")
