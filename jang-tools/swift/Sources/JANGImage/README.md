# JANGImage

Native Swift port of `jang_tools.vl.pixtral`. Replaces numpy + PIL with
CoreGraphics (image decode + resample) and a SIMD-friendly normalize loop.

Used by Mistral-Medium-3.5-128B (mistral3 + pixtral) and any other
pixtral-style VL JANG ships. The Python reference stays the source of truth
for correctness; this Swift target is the production-runtime path.

## Tests

Add an XCTest pair for any image you suspect rounding-mode drift on:
```swift
let (chw, h, w) = try PixtralImageProcessor().preprocess(cgImage: img)
```
then compare element-wise to the Python output (load via NPZ).
