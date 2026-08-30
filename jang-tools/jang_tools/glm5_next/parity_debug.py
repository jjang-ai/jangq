"""Layer-wise parity bisection vs the torch reference (tiny model)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/tmp/glm5tf")
import numpy as np

from jang_tools.glm5_next.parity_tiny import build_torch


def main():
    import torch
    tmp = Path(tempfile.mkdtemp(prefix="glm5dbg_"))
    ids_np, _ = build_torch(tmp)

    # torch per-layer streams
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    cfg = Glm5NextTextConfig(**json.loads((tmp / "config.json").read_text())["text_config"])
    torch.manual_seed(0)
    tm = Glm5NextTextModel(cfg).eval().float()
    ids = torch.tensor(ids_np)
    with torch.no_grad():
        out = tm(input_ids=ids, use_cache=False, output_hidden_states=True)
    t_layers = [h.numpy() for h in out.hidden_states]  # embeds + after each layer

    import mlx.core as mx
    from jang_tools.glm5_next.load import load_model
    model = load_model(str(tmp), dtype=mx.float32)

    x = model.model.embed_tokens(mx.array(ids_np))
    streams = mx.broadcast_to(x[:, :, None, :], (*x.shape[:2], 4, x.shape[-1]))
    print("embeds Δ:", np.abs(np.asarray(streams) - t_layers[0]).max()
          if t_layers[0].ndim == 4 else "(torch records collapsed?) shape " + str(t_layers[0].shape))
    for i, layer in enumerate(model.model.layers):
        # sub-component probes on the first divergent layer path
        post, comb, collapsed = layer.attn_hc(streams)
        streams = layer(streams)
        d = np.abs(np.asarray(streams) - t_layers[i + 1]).max()
        print(f"after layer {i} ({'KDA' if layer.is_linear else 'MLA'}"
              f"+{'dense' if not hasattr(layer.mlp, 'switch_mlp') else 'moe'}): maxΔ={d:.3e}")
    # torch attn_hc internals for layer0
    with torch.no_grad():
        h0 = torch.tensor(t_layers[0])
        tpost, tcomb, tcoll = tm.layers[0].attn_hc(h0)
    x0 = mx.array(t_layers[0])
    p2, c2, col2 = model.model.layers[0].attn_hc(x0)
    print("hc post Δ:", np.abs(np.asarray(p2) - tpost.numpy()).max())
    print("hc comb Δ:", np.abs(np.asarray(c2) - tcomb.numpy()).max())
    print("hc coll Δ:", np.abs(np.asarray(col2) - tcoll.numpy()).max())
    # attn output on identical input
    with torch.no_grad():
        tattn = tm.layers[0].self_attn(tm.layers[0].input_layernorm(tcoll))
    mattn = model.model.layers[0].self_attn(model.model.layers[0].input_layernorm(col2))
    print("layer0 KDA out Δ:", np.abs(np.asarray(mattn) - tattn.numpy()).max())
    # dense mlp
    with torch.no_grad():
        tmlp = tm.layers[0].mlp(tm.layers[0].post_attention_layernorm(tcoll))
    mmlp = model.model.layers[0].mlp(model.model.layers[0].post_attention_layernorm(col2))
    print("layer0 MLP out Δ:", np.abs(np.asarray(mmlp) - tmlp.numpy()).max())


if __name__ == "__main__":
    main()
