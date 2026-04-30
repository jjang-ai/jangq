//
//  JANGCxx.h — C++ shim used by JANGRuntime when calling into native
//  TurboQuant matmul kernels (Metal-backed in vmlx-swift).
//
//  This header gives Swift a stable C ABI so the JANGRuntime target can
//  keep its pure-Swift surface and import the kernel bridge through this
//  module.
//

#ifndef JANGCXX_H
#define JANGCXX_H

#ifdef __cplusplus
extern "C" {
#endif

// Decode a JANGTQ-packed expert tile to bf16. `packed` is uint8 indices
// into `codebook`; `out` receives bf16 results. Group size is fixed at 64
// for now; if you need a different size pass it through the Python convert
// step.
//
// In production this delegates to the Metal kernel in vmlx-swift's
// vMLXLMCommon TurboQuant family. The shim is here so JANGRuntime can
// link against a stable Swift package without inheriting that dependency.
void jang_tq_decode_bf16(const unsigned char *packed,
                         const unsigned short *codebook_bf16,
                         int n_rows, int n_cols, int group_size, int bits,
                         unsigned short *out_bf16);

// Pixtral image preprocess — native Swift wrapper sits in JANGImage. This
// header only declares the matmul shim.

#ifdef __cplusplus
}
#endif

#endif
