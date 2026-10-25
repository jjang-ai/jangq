"""Raw standalone prefill pp/s for the DSV4.1 runtime (no engine/scheduler overhead).
Times m.forward(prompt) for several sizes, warm (2nd run), to find the runtime prefill ceiling."""
import sys, time
from pathlib import Path
import mlx.core as mx
sys.path.insert(0, str(Path.home() / "jang/jang-tools"))
from jang_tools.dsv41.model import DSV41

m = DSV41(str(Path.home() / "models/DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX"))
vocab = m.args.vocab_size
import random
for n in (512, 1024, 2048, 4096):
    ids = [random.randint(1000, min(vocab - 1, 100000)) for _ in range(n)]
    # natural-ish: repeat a small coherent token window so engram rows are cache-coherent
    base = m.tokenizer.encode("The quick brown fox jumps over the lazy dog. ", add_special_tokens=False).ids
    ids = (base * (n // len(base) + 1))[:n]
    best = None
    for trial in range(2):
        sess = m.new_session()
        t0 = time.time()
        logits = m.forward(ids, sess)
        mx.eval(logits)
        dt = time.time() - t0
        pps = n / dt
        best = pps if best is None else max(best, pps)
        del sess, logits; mx.clear_cache()
    print(f"n={n:5d}  prefill {best:7.1f} pp/s  ({n/best*1000:.0f} ms)", flush=True)
