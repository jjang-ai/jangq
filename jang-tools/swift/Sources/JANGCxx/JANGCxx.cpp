//
//  JANGCxx — placeholder body. Real implementation links against
//  vmlx-swift's TurboQuant Metal kernel. This file exists so the Swift
//  package compiles; once vmlx exposes a public C entry point, swap the
//  body below for a call to that.
//

#include "JANGCxx.h"

extern "C" void jang_tq_decode_bf16(
    const unsigned char *packed,
    const unsigned short *codebook_bf16,
    int n_rows, int n_cols, int group_size, int bits,
    unsigned short *out_bf16
) {
    // Stub: zero-fill. Production binds to the Metal kernel.
    int total = n_rows * n_cols;
    for (int i = 0; i < total; ++i) out_bf16[i] = 0;
    (void)packed; (void)codebook_bf16; (void)group_size; (void)bits;
}
