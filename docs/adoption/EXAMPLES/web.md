# Web / WebAssembly JANG inference

## Status: not yet supported

As of v1.0, JANG requires Apple Silicon (Metal). Browser-based inference via WebGPU or
WebAssembly is not yet implemented.

## Why not yet

JANG's mixed-precision block format requires custom dequantization kernels. The reference
Metal kernels (`JangV2QuantMatmul.metal`, `JANGTQMatmul.metal`) are Apple-specific.
Porting them to WebGPU is tracked as a post-v1 goal.

The on-disk format itself is portable — it is standard safetensors with a bit-packing
convention documented in [PORTING.md](../PORTING.md). The blocker is a fused
unpack-dequant-matmul kernel that runs fast enough in a browser context.

## If you want to help

1. Read [PORTING.md](../PORTING.md) — the format is fully documented there.
2. The Metal kernels at `jang-runtime/Sources/JANGCoreMetal/*.metal` are the reference
   implementation. The core operation is a fused unpack-dequant-matmul.
3. A WebGPU port would target WGSL shaders. A WASM-only port would use a software
   dequant loop + WASM-SIMD matmul. Either path would need to handle the codebook
   lookup step for JANGTQ expert weights.

Open an issue at https://github.com/jjang-ai/jangq/issues with the `web` label if you
are interested in driving this.

## Fallback today: server-side JANG + JavaScript client

Until native web inference lands, the practical path is:

1. Serve the JANG model via `vmlx-engine` on a Mac (see [server.md](server.md))
2. Call it from your web client via the OpenAI-compatible HTTP API

This gives you browser-delivered inference today without a WebGPU or WASM runtime.

```javascript
// Calling a local vmlx-engine server from a browser or Node client
const response = await fetch("http://127.0.0.1:8000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "default",
    messages: [{ role: "user", content: "Hello" }],
    stream: false,
  }),
});

const data = await response.json();
console.log(data.choices[0].message.content);
```
