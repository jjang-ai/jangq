"""Research GPTQ with quantized-prefix and quantized gate/up calibration.

Streams one decoder layer from a verified mixed-precision bundle. Non-expert
weights and native MXFP4 projections remain exactly those of that bundle.
Each affine projection selects among the incumbent, fitted RTN and GPTQ codes
using its new calibration Hessian; this is not a held-out quality guarantee.
No model bundle is modified. Outputs are converter-compatible projection files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import _gather_sort
from .convert_v26 import load_plan
from .v26_source import SourceStream, layer_masks
from .v26_model import DecoderLayer, ModelArgs
from .v26_gptq import _seg_hessians, gptq_round, h_err
from .v26_gptq_provenance import bind_run, file_sha256
from .v26_quant import fit_affine, pack_codes
from .v26_sweep import awq_scale, expert_importance
from .mxfp4_codec import mxfp4_raw_to_mlx


class BundleStream:
    """Layer reader using the bundle's real overrides and folded AWQ tensors."""
    def __init__(self, bundle):
        self.bundle = Path(bundle)
        self.config = json.loads((self.bundle / 'config.json').read_text())
        self.index = json.loads((self.bundle / 'model.safetensors.index.json').read_text())['weight_map']
        self.quant = self.config['quantization']
        self.args = ModelArgs(**{k:v for k,v in self.config.items() if k in ModelArgs.__dataclass_fields__})

    def tensors(self, prefix):
        names = [k for k in self.index if k.startswith(prefix)]
        out = {}
        for fn in sorted({self.index[k] for k in names}):
            values = mx.load(str(self.bundle / fn))
            out.update({k[len(prefix):]:values[k] for k in names if self.index[k] == fn})
            del values
        return out

    def layer(self, L):
        layer = DecoderLayer(self.args, L)
        prefix = f'model.layers.{L}.'
        default = {k:self.quant[k] for k in ('bits','group_size','mode') if k in self.quant}
        def predicate(path, module):
            return self.quant.get(prefix + path, default) if hasattr(module, 'to_quantized') else False
        nn.quantize(layer, class_predicate=predicate)
        values = self.tensors(prefix)
        expected = {k for k,_ in tree_flatten(layer.parameters())}
        if expected != set(values):
            raise ValueError(f'L{L} tensor mismatch: missing={expected-set(values)}, extra={set(values)-expected}')
        layer.load_weights(list(values.items()), strict=True)
        mx.eval(layer.parameters())
        return layer

    def embed(self, ids):
        spec = self.quant.get('model.embed_tokens', self.quant)
        emb = nn.QuantizedEmbedding(self.args.vocab_size, self.args.hidden_size,
                                    group_size=spec['group_size'], bits=spec['bits'])
        emb.load_weights(list(self.tensors('model.embed_tokens.').items()), strict=True)
        result = [emb(mx.array(b)) for b in ids]
        mx.eval(result)
        return result


def install(layer, L, proj, directory):
    path = directory / f'L{L}.{proj}.safetensors'
    if not path.exists():
        return
    values, meta = mx.load(str(path), return_metadata=True)
    module = getattr(layer.mlp.switch_mlp, proj)
    if meta['bits'] != str(module.bits) or meta['group_size'] != str(module.group_size):
        raise ValueError(f'Overlay precision mismatch {path}')
    for key, value in values.items():
        old = getattr(module, key)
        if old.shape != value.shape or old.dtype != value.dtype:
            raise ValueError(f'Overlay tensor mismatch {path}:{key}')
    module.update(values)
    mx.eval(module.parameters())


def solve(ss, L, proj, spec, hessian, scale, importance, incumbent, out, report):
    E = ss.args.n_routed_experts
    bits, gs = spec['bits'], spec['group_size']
    packed, scales, biases = [], [], []
    selections = np.zeros(3, dtype=np.int64)
    ratios = []
    for first in range(0, E, 32):
        last = min(first + 32, E)
        raw = [mxfp4_raw_to_mlx(*ss.idx.read_mxfp4_raw(f'model.layers.{L}.mlp.experts.{e}.{proj}.weight'))
               for e in range(first, last)]
        W = mx.dequantize(mx.array(np.stack([r[0] for r in raw])),
                         mx.array(np.stack([r[1] for r in raw])), group_size=32, bits=4, mode='mxfp4').astype(mx.float32)
        if scale is not None and proj != 'down_proj':
            W = W * scale
        fp, fs, fb = fit_affine(W, importance[first:last], bits=bits, group_size=gs)
        H = hessian(first, last)
        ip, isc, ib = incumbent.weight[first:last], incumbent.scales[first:last], incumbent.biases[first:last]
        # This trial changes calibration inputs only, not the affine grid.
        if not bool(mx.array_equal(fs, isc).item()) or not bool(mx.array_equal(fb, ib).item()):
            raise ValueError(f'Incumbent grid differs from fixed recipe L{L}.{proj}')
        mx.eval(W, H, fp, fs, fb)
        codes = []
        for e in range(0, last-first, 4):
            codes.append(gptq_round(W[e:e+4], H[e:e+4], fs[e:e+4], fb[e:e+4], bits, gs))
            mx.eval(codes[-1])
        gp = pack_codes(mx.concatenate(codes), bits)
        options = [ip, fp, gp]
        errors = mx.stack([h_err(mx.dequantize(p, fs, fb, bits=bits, group_size=gs).astype(mx.float32), W, H)
                           for p in options])
        choice = mx.argmin(errors, axis=0)  # tie preserves incumbent
        selected = mx.where((choice == 2)[:,None,None], gp,
                            mx.where((choice == 1)[:,None,None], fp, ip))
        best = mx.min(errors, axis=0)
        mx.eval(selected, errors, choice)
        ev = np.array(errors); bv = np.array(best)
        if not np.isfinite(ev).all() or not np.all(bv <= ev[0] + np.maximum(abs(ev[0])*1e-6, 1e-8)):
            raise ValueError('Invalid reconstruction selection')
        selections += np.bincount(np.array(choice), minlength=3)
        ratios.extend((1-bv/np.maximum(ev[0],1e-30)).tolist())
        packed.append(selected); scales.append(fs); biases.append(fb)
        print(f'[seq-gptq] L{L}.{proj} experts {first}:{last} selected {selections.tolist()}', flush=True)
    tensors = {'weight':mx.concatenate(packed), 'scales':mx.concatenate(scales), 'biases':mx.concatenate(biases)}
    path = out / f'L{L}.{proj}.safetensors'
    tmp = out / f'L{L}.{proj}.pending.safetensors'
    mx.save_safetensors(str(tmp), tensors, metadata={'bits':str(bits),'group_size':str(gs),'awq':str(scale is not None)})
    tmp.replace(path)
    report[f'L{L}.{proj}'] = {'spec':spec, 'experts':E, 'selection_counts':dict(zip(['incumbent','fit_rtn','sequential_gptq'], selections.tolist())),
        'mean_herr_reduction_vs_incumbent':float(np.mean(ratios)), 'median_herr_reduction_vs_incumbent':float(np.median(ratios)), 'sha256':file_sha256(path)}
    (out/'gptq_report.json').write_text(json.dumps(report,indent=2)+'\n')


def bind_sequential(out, bundle, tokens, plan, src, limit_layers):
    bind_run(out, src, tokens, plan, tau_scale=1.0, damp=0.01)
    names = ['v26_sequential_gptq.py','v26_gptq.py','v26_quant.py','v26_model.py','v26_source.py','v26_sweep.py']
    expected = {'schema':'mimo-v26-sequential-gptq-v1', 'propagation':'actual quantized bundle prefix; current quantized gate/up before down capture',
        'incumbent_manifest_sha256':file_sha256(bundle/'SHA256-MANIFEST.json'),
        'source_files_sha256':{n:file_sha256(Path(__file__).with_name(n)) for n in names},
        'max_layers':limit_layers, 'tokens_sha256':hashlib.sha256(tokens.tobytes()).hexdigest(),
        'grid':'fixed imatrix BF16', 'selection':'per-expert minimum calibration Hessian error among incumbent, fitted RTN, new GPTQ'}
    p = out/'sequential_run.json'
    if p.exists() and json.loads(p.read_text()) != expected:
        raise ValueError('Sequential resume provenance changed; use a new directory')
    if not p.exists() and any(out.glob('L*.safetensors')):
        raise ValueError('Unbound sequential outputs')
    p.write_text(json.dumps(expected,indent=2)+'\n')


def run(src, bundle, tokens, plan_path, out, max_layers=0):
    raw_plan = json.loads(plan_path.read_text())
    bind_sequential(out, bundle, tokens, raw_plan, src, max_layers)
    ss, bs = SourceStream(src), BundleStream(bundle)
    a = ss.args
    plan = load_plan(plan_path, a.num_hidden_layers, [L for L in range(a.num_hidden_layers) if a.moe_layer_freq[L]])
    stats = mx.load(plan['stats'])
    alpha = {int(k):float(v) for k,v in (plan.get('awq_alpha') or {}).items()}
    real = np.array([np.flatnonzero(r != 151643)[-1]+1 if np.any(r!=151643) else 0 for r in tokens])
    if np.any(real == 0):raise ValueError('Empty calibration row')
    order = np.argsort(-real)
    batches = [tokens[i:i+1,:real[i]] for i in order]
    hs = bs.embed(batches)
    report_path = out/'gptq_report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    capture_path = out/'hessian_capture_report.json'
    capture = json.loads(capture_path.read_text()) if capture_path.exists() else {}
    masks = {}; start = time.time()
    for L in range(max_layers or a.num_hidden_layers):
        layer = bs.layer(L)
        moe = bool(a.moe_layer_freq[L])
        affine = [p for p in ('gate_proj','up_proj','down_proj') if moe and plan['_units'][(L,p)]['mode']=='affine']
        for p in affine:
            path=out/f'L{L}.{p}.safetensors'
            if path.exists():
                if f'L{L}.{p}' not in report or file_sha256(path)!=report[f'L{L}.{p}']['sha256']:
                    raise ValueError('Incomplete or modified projection checkpoint')
                install(layer,L,p,out)
        todo = [p for p in affine if not (out/f'L{L}.{p}.safetensors').exists()]
        scale = awq_scale(stats,L,alpha[L]) if alpha.get(L) else None
        row = capture.get(str(L), {'valid_tokens':int(real.sum()), 'phases':{}})
        # Quantized attention/non-expert path is the exact bundle path, including AWQ folding.
        residuals=[]
        for h in hs:
            T=h.shape[1]
            if T not in masks:masks[T]=layer_masks(a,T)
            mask=masks[T][1 if layer.is_swa else 0]
            residuals.append(h+layer.self_attn(layer.input_layernorm(h),mask));mx.eval(residuals[-1])
        del hs
        for phase, projs, dim in [('gate_up',[p for p in todo if p!='down_proj'],a.hidden_size),
                                   ('down',[p for p in todo if p=='down_proj'],a.moe_intermediate_size)]:
            if not projs:continue
            Hs=[mx.zeros((dim,dim)) for _ in range(a.n_routed_experts)];counts=np.zeros(a.n_routed_experts)
            for i,h in enumerate(residuals):
                x=layer.post_attention_layernorm(h).reshape(-1,a.hidden_size)
                idx,_=layer.mlp.gate(x);xs,ids,_=_gather_sort(mx.expand_dims(x,(-2,-3)),idx)
                if phase=='down':
                    sw=layer.mlp.switch_mlp
                    xs=nn.silu(sw.gate_proj(xs,ids,sorted_indices=True))*sw.up_proj(xs,ids,sorted_indices=True)
                _seg_hessians(xs.reshape(-1,dim),ids,a.n_routed_experts,Hs,counts)
                print(f'[seq-gptq] capture L{L}.{phase} {i+1}/{len(residuals)}',flush=True)
            if int(counts.sum())!=int(real.sum())*a.num_experts_per_tok:raise ValueError('Routed-token coverage mismatch')
            pool=sum(Hs)/max(float(counts.sum()),1.0);mx.eval(pool)
            def hessian(first,last):
                H=mx.stack([(Hs[e]+dim*pool)/(float(counts[e])+dim) for e in range(first,last)])
                diagonal=mx.diagonal(H,axis1=-2,axis2=-1)
                return H+(0.01*diagonal.mean(-1)+1e-8)[:,None,None]*mx.eye(dim)[None]
            row['phases'][phase]={'routed_tokens':int(counts.sum()),'experts_with_tokens':int((counts>0).sum()),
                'expert_min':int(counts.min()),'expert_max':int(counts.max()),'input_coordinates':'AWQ-folded native bundle' if phase=='gate_up' else 'newly selected quantized gate/up'}
            for p in projs:
                solve(ss,L,p,plan['_units'][(L,p)],hessian,scale,expert_importance(stats,L,p,scale),getattr(layer.mlp.switch_mlp,p),out,report)
                install(layer,L,p,out)
            del hessian,Hs,pool;mx.clear_cache()
        hs=[]
        for h in residuals:
            hs.append(h+layer.mlp(layer.post_attention_layernorm(h)));mx.eval(hs[-1])
        row['completed']=True;row['peak_gib']=mx.get_peak_memory()/2**30
        row['cumulative_seconds']=time.time()-start;capture[str(L)]=row
        capture_path.write_text(json.dumps(capture,indent=2)+'\n')
        del layer,residuals;mx.clear_cache()
        print(f'[seq-gptq] layer {L} done {time.time()-start:.0f}s peak {row["peak_gib"]:.2f}GiB',flush=True)
    print('[seq-gptq] DONE' if not max_layers else '[seq-gptq] SMOKE DONE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--src',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--tokens',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--max-layers',type=int,default=0)
    a=p.parse_args();tokens=np.load(a.tokens)
    if a.limit:tokens=tokens[:a.limit]
    run(a.src,a.bundle,tokens,a.plan,a.out,a.max_layers)

if __name__=='__main__':main()
