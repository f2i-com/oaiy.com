//! CUDA kernel source. Compiled at runtime via NVRTC.
//!
//! All kernels are F32 only and unoptimized — naive correct implementations
//! to validate the backend wiring. Real perf comes from later milestones:
//! cuBLAS dispatch for `linear`, fused attention, on-device tensors so we
//! don't shuttle every op through host memory.

pub const KERNEL_SRC: &str = r#"
// NVRTC doesn't include <math.h> by default, so we re-define a few constants.
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// f16 -> f32 by hand. NVRTC can't always be relied on to surface
// __half2float; this is portable across all compute capabilities.
__device__ __forceinline__ float f16_to_f32(unsigned short bits) {
    unsigned int sign = (unsigned int)(bits >> 15) & 1u;
    unsigned int exp  = (unsigned int)(bits >> 10) & 0x1Fu;
    unsigned int mant = (unsigned int)(bits) & 0x3FFu;
    unsigned int out;
    if (exp == 0u) {
        if (mant == 0u) {
            out = sign << 31;
        } else {
            // Subnormal — normalize.
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400u) == 0u);
            mant &= 0x3FFu;
            unsigned int e32 = (unsigned int)((int)112 - e);
            out = (sign << 31) | (e32 << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        out = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        out = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    return __uint_as_float(out);
}

extern "C" {

// Cooperative warp GEMV for dense F32, decode-time M=1 fast path. Sits below
// the cuBLAS dispatch threshold (100M FLOPs) and beats the naive 16x16 kernel
// at small N — cuBLAS has setup cost; the naive kernel does only 1 productive
// thread per warp at M=1. This kernel uses 32 threads per warp to partition K
// (each thread does K/32 contiguous elements), with the standard butterfly
// shfl-xor reduction. Block packs 8 warps = 8 rows. Used by Qwen3.6 27B's F32
// ssm_ba [96,5120] decode matmul (was the slowest single op per token).
__global__ void linear_f32_gemv_coop(const float* __restrict__ x,
                                      const float* __restrict__ w,
                                      float* __restrict__ y,
                                      int N, int K) {
    int n = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= N) return;
    int t = threadIdx.x;          // [0, 32)
    int per_thd = K / 32;         // requires K % 32 == 0
    int start = t * per_thd;
    int end = start + per_thd;
    const float* w_row = w + (long)n * K;

    float acc = 0.0f;
    for (int k = start; k < end; ++k) {
        acc += x[k] * w_row[k];
    }

    // Warp-reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (t == 0) y[n] = acc;
}

// Cooperative warp GEMV for Q8_0, decode-time M=1 fast path. Different design
// from the K-quants coop kernels: each thread handles ONE element-position
// within every block (thread t works on element t of every block, looping over
// blocks). Adjacent threads in the warp read adjacent qs[] bytes within the
// same block — perfectly coalesced. Works for any K%32==0 (every Q8_0 weight
// satisfies this by construction). Block geometry: blockDim=(32, 8, 1) packs
// 8 output rows per block.
__global__ void linear_q8_0_gemv_coop_f32(const float* __restrict__ x,
                                           const unsigned char* __restrict__ w,
                                           float* __restrict__ y,
                                           int N, int K) {
    int n = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= N) return;
    int t = threadIdx.x;          // [0, 32) — thread handles element t within each block
    int n_blocks  = K / 32;
    int row_bytes = n_blocks * 34;
    const unsigned char* w_row = w + (long)n * row_bytes;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 34;
        unsigned short scale_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        float d = f16_to_f32(scale_bits);
        const signed char* qs = (const signed char*)(bp + 2);
        const float* xb = x + b * 32;
        acc += xb[t] * (float)qs[t] * d;
    }

    // Warp-reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (t == 0) y[n] = acc;
}

// linear with Q8_0 packed weights. x:[M, K] f32; w:[N, K] packed Q8_0; y:[M, N] f32.
// Q8_0 block layout (32 elements per block, 34 bytes / block):
//   bytes [0..2]  f16 scale d
//   bytes [2..34] i8  qs[32]   --> dequant value = qs[i] * d
// W is laid out as N rows × (K/32) blocks × 34 bytes.
__global__ void linear_q8_0_f32(const float* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 float* __restrict__ y,
                                 int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 32;
    int row_bytes = n_blocks * 34;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 34;
        unsigned short scale_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        float d = f16_to_f32(scale_bits);
        const signed char* qs = (const signed char*)(bp + 2);
        const float* xb = x_row + b * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            acc += xb[i] * (float)qs[i] * d;
        }
    }
    y[(long)m * N + n] = acc;
}

// linear with Q4_K packed weights. x:[M, K] f32; w:[N, K] packed Q4_K; y:[M, N] f32.
// Q4_K super-block (256 elements per block, 144 bytes):
//   bytes [0..2]    f16  d
//   bytes [2..4]    f16  dmin
//   bytes [4..16]   u8   scales[12]   (8 6-bit scales + 8 6-bit mins, packed)
//   bytes [16..144] u8   qs[128]      (256 4-bit nibbles)
// Per-row layout: N rows × (K/256) blocks × 144 bytes.
__device__ __forceinline__ void q4k_unpack_scale_min(int j, const unsigned char* scales,
                                                      unsigned char* d_out,
                                                      unsigned char* m_out) {
    if (j < 4) {
        *d_out = scales[j] & 0x3F;
        *m_out = scales[j + 4] & 0x3F;
    } else {
        *d_out = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m_out = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4);
    }
}
// Cooperative warp GEMV for Q4_K, decode-time M=1 fast path. Each warp (32
// threads) computes ONE output element by partitioning the K-axis: thread t
// handles K/32 contiguous elements (= K/1024 contiguous sub-blocks). A final
// __shfl_xor_sync reduction combines per-thread partial sums. Adjacent threads
// thus read adjacent regions of the same row, producing coalesced loads —
// the existing per-thread-one-row kernel can't get this since its threads
// scatter across distinct rows. Requires K % 1024 == 0 (true for our 4096,
// 8192, 12288 hidden / ff dims). Block shape: (32, OUT_PER_BLOCK, 1) — each
// warp handles one output, OUT_PER_BLOCK warps per block.
__global__ void linear_q4_k_gemv_coop_f32(const float* __restrict__ x,
                                           const unsigned char* __restrict__ w,
                                           float* __restrict__ y,
                                           int N, int K) {
    int n = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= N) return;
    int t = threadIdx.x;          // [0, 32)
    int n_blocks    = K / 256;
    int total_sb    = n_blocks * 8;
    int sb_per_thd  = total_sb / 32;     // 4 / 8 / 12 for K = 4096 / 8192 / 12288
    int sb_start    = t * sb_per_thd;
    int sb_end      = sb_start + sb_per_thd;
    int row_bytes   = n_blocks * 144;
    const unsigned char* w_row = w + (long)n * row_bytes;

    // Cache the most recent superblock header across consecutive sub-blocks.
    int last_b = -1;
    float d_sb = 0.0f, min_sb = 0.0f;
    const unsigned char* qs_sb = nullptr;
    const unsigned char* sc_sb = nullptr;

    float acc = 0.0f;
    for (int sb = sb_start; sb < sb_end; ++sb) {
        int b  = sb >> 3;             // / 8
        int is = sb & 7;              // % 8
        if (b != last_b) {
            const unsigned char* bp = w_row + b * 144;
            unsigned short d_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
            unsigned short m_bits = (unsigned short)bp[2] | ((unsigned short)bp[3] << 8);
            d_sb   = f16_to_f32(d_bits);
            min_sb = f16_to_f32(m_bits);
            sc_sb  = bp + 4;
            qs_sb  = bp + 16;
            last_b = b;
        }
        unsigned char sc_byte, mm_byte;
        q4k_unpack_scale_min(is, sc_sb, &sc_byte, &mm_byte);
        float dq  = d_sb   * (float)sc_byte;
        float mq  = min_sb * (float)mm_byte;
        // sub-block `is` covers elements [is*32..(is+1)*32) of the superblock.
        // Quantized bytes for this sub-block: qs_sb[(is/2)*32..(is/2)*32+32].
        // Even is → low nibbles, odd is → high nibbles.
        int qs_off = (is >> 1) * 32;
        int x_off  = b * 256 + is * 32;
        const unsigned char* qsp = qs_sb + qs_off;
        const float* xp = x + x_off;
        if ((is & 1) == 0) {
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                float qv = (float)(qsp[l] & 0x0F);
                acc += xp[l] * (dq * qv - mq);
            }
        } else {
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                float qv = (float)((qsp[l] >> 4) & 0x0F);
                acc += xp[l] * (dq * qv - mq);
            }
        }
    }

    // Warp-reduce the partial sums into thread 0.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (t == 0) y[n] = acc;
}

__global__ void linear_q4_k_f32(const float* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 float* __restrict__ y,
                                 int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 256;
    int row_bytes = n_blocks * 144;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 144;
        unsigned short d_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        unsigned short m_bits = (unsigned short)bp[2] | ((unsigned short)bp[3] << 8);
        float d   = f16_to_f32(d_bits);
        float min = f16_to_f32(m_bits);
        const unsigned char* scales = bp + 4;
        const unsigned char* qs     = bp + 16;
        const float* xb = x_row + b * 256;

        // 8 sub-blocks of 32 elements each, but the conversion from
        // dequant_q4_K loops 4 outer iters of 64 elems with 2 sub-blocks per iter.
        int q_off = 0;
        int y_off = 0;
        for (int is = 0; is < 8; is += 2) {
            unsigned char sc1, mm1, sc2, mm2;
            q4k_unpack_scale_min(is + 0, scales, &sc1, &mm1);
            q4k_unpack_scale_min(is + 1, scales, &sc2, &mm2);
            float d1 = d * (float)sc1; float min1 = min * (float)mm1;
            float d2 = d * (float)sc2; float min2 = min * (float)mm2;
            // Low nibbles: 32 elems with sub-block scale `is`.
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                float qv = (float)(qs[q_off + l] & 0x0F);
                acc += xb[y_off + l] * (d1 * qv - min1);
            }
            // High nibbles: 32 elems with sub-block scale `is+1`.
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                float qv = (float)((qs[q_off + l] >> 4) & 0x0F);
                acc += xb[y_off + 32 + l] * (d2 * qv - min2);
            }
            q_off += 32;
            y_off += 64;
        }
    }
    y[(long)m * N + n] = acc;
}

// Cooperative warp GEMV for Q6_K, decode-time M=1 fast path. Same pattern as
// the Q4_K coop kernel but partitions WITHIN each super-block instead of across
// super-blocks (Q6_K has only n_blocks=K/256 of them, often < 32 for lm_head
// where K=hidden_dim and is small). Each warp computes ONE output element;
// thread t (0..31) handles inner-step l=t in BOTH outer iters of every
// super-block — 8 multiply-adds per super-block per thread. Final shfl-xor
// butterfly reduces across the warp. Requires K % 256 == 0 (every K-quant
// model satisfies this by construction since the super-block is 256-wide).
__global__ void linear_q6_k_gemv_coop_f32(const float* __restrict__ x,
                                           const unsigned char* __restrict__ w,
                                           float* __restrict__ y,
                                           int N, int K) {
    int n = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= N) return;
    int t = threadIdx.x;          // [0, 32) — corresponds to inner step l
    int n_blocks  = K / 256;
    int row_bytes = n_blocks * 210;
    const unsigned char* w_row = w + (long)n * row_bytes;

    int l = t;
    int is = l >> 4;              // 0 (l<16) or 1 (l>=16)

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 210;
        const unsigned char* ql = bp + 0;
        const unsigned char* qh = bp + 128;
        const signed char*   sc = (const signed char*)(bp + 192);
        unsigned short d_bits = (unsigned short)bp[208] | ((unsigned short)bp[209] << 8);
        float d = f16_to_f32(d_bits);
        const float* xb = x + b * 256;

        // Outer iter 0
        {
            int q1 = (int)((ql[l]      & 0x0F) | (((qh[l] >> 0) & 0x3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x3) << 4)) - 32;
            int q3 = (int)((ql[l]      >> 4)   | (((qh[l] >> 4) & 0x3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 0x3) << 4)) - 32;
            acc += xb[l]      * d * (float)sc[is + 0] * (float)q1;
            acc += xb[l + 32] * d * (float)sc[is + 2] * (float)q2;
            acc += xb[l + 64] * d * (float)sc[is + 4] * (float)q3;
            acc += xb[l + 96] * d * (float)sc[is + 6] * (float)q4;
        }
        // Outer iter 1: ql_off=64, qh_off=32, sc_off=8, xb_off=128
        {
            int q1 = (int)((ql[64 + l]      & 0x0F) | (((qh[32 + l] >> 0) & 0x3) << 4)) - 32;
            int q2 = (int)((ql[64 + l + 32] & 0x0F) | (((qh[32 + l] >> 2) & 0x3) << 4)) - 32;
            int q3 = (int)((ql[64 + l]      >> 4)   | (((qh[32 + l] >> 4) & 0x3) << 4)) - 32;
            int q4 = (int)((ql[64 + l + 32] >> 4)   | (((qh[32 + l] >> 6) & 0x3) << 4)) - 32;
            acc += xb[128 + l]      * d * (float)sc[8 + is + 0] * (float)q1;
            acc += xb[128 + l + 32] * d * (float)sc[8 + is + 2] * (float)q2;
            acc += xb[128 + l + 64] * d * (float)sc[8 + is + 4] * (float)q3;
            acc += xb[128 + l + 96] * d * (float)sc[8 + is + 6] * (float)q4;
        }
    }

    // Warp-reduce the partial sums into thread 0.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (t == 0) y[n] = acc;
}

// linear with Q6_K packed weights. x:[M, K] f32; w:[N, K] packed Q6_K; y:[M, N] f32.
// Q6_K super-block (256 elements per block, 210 bytes):
//   bytes [0..128]   u8   ql[128]      (low 4 bits of each value)
//   bytes [128..192] u8   qh[64]       (high 2 bits, packed 4 per byte)
//   bytes [192..208] i8   scales[16]   (one signed scale per 16-elem group)
//   bytes [208..210] f16  d            (super-block scale)
// Each value is `(low4 | (high2 << 4)) - 32` × `d` × `scales[group]`.
__global__ void linear_q6_k_f32(const float* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 float* __restrict__ y,
                                 int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 256;
    int row_bytes = n_blocks * 210;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 210;
        const unsigned char* ql = bp + 0;
        const unsigned char* qh = bp + 128;
        const signed char*   sc = (const signed char*)(bp + 192);
        unsigned short d_bits = (unsigned short)bp[208] | ((unsigned short)bp[209] << 8);
        float d = f16_to_f32(d_bits);
        const float* xb = x_row + b * 256;

        // Two outer iters of 128 elems each. Inner: 32 threads' worth of work,
        // executed as a serial loop in this per-output-element kernel.
        int xb_off = 0;
        int ql_off = 0;
        int qh_off = 0;
        int sc_off = 0;
        for (int it = 0; it < 2; ++it) {
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = (int)((ql[ql_off + l]      & 0x0F) | (((qh[qh_off + l] >> 0) & 0x3) << 4)) - 32;
                int q2 = (int)((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 0x3) << 4)) - 32;
                int q3 = (int)((ql[ql_off + l]      >> 4)   | (((qh[qh_off + l] >> 4) & 0x3) << 4)) - 32;
                int q4 = (int)((ql[ql_off + l + 32] >> 4)   | (((qh[qh_off + l] >> 6) & 0x3) << 4)) - 32;
                acc += xb[xb_off + l]      * d * (float)sc[sc_off + is + 0] * (float)q1;
                acc += xb[xb_off + l + 32] * d * (float)sc[sc_off + is + 2] * (float)q2;
                acc += xb[xb_off + l + 64] * d * (float)sc[sc_off + is + 4] * (float)q3;
                acc += xb[xb_off + l + 96] * d * (float)sc[sc_off + is + 6] * (float)q4;
            }
            xb_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    y[(long)m * N + n] = acc;
}

// linear with Q2_K packed weights. x:[M, K] f32; w:[N, K] packed Q2_K; y:[M, N] f32.
// Q2_K super-block (256 elements per block, 84 bytes):
//   bytes [0..16]   u8  scales[16]   -- 16 4-bit scales (low nibble) + 16 4-bit mins (high nibble)
//   bytes [16..80]  u8  qs[64]       -- 256 2-bit quants, 4 values per byte
//   bytes [80..82]  f16 d
//   bytes [82..84]  f16 dmin
__global__ void linear_q2_k_f32(const float* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 float* __restrict__ y,
                                 int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 256;
    int row_bytes = n_blocks * 84;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp     = w_row + b * 84;
        const unsigned char* scales = bp + 0;
        const unsigned char* qs     = bp + 16;
        unsigned short d_bits   = (unsigned short)bp[80] | ((unsigned short)bp[81] << 8);
        unsigned short m_bits   = (unsigned short)bp[82] | ((unsigned short)bp[83] << 8);
        float d   = f16_to_f32(d_bits);
        float dmin = f16_to_f32(m_bits);
        const float* xb = x_row + b * 256;

        int y_off = 0;
        int q_base = 0;
        int is = 0;
        for (int outer = 0; outer < 2; ++outer) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                unsigned char sc1 = scales[is++];
                float dl1 = d * (float)(sc1 & 0x0F);
                float ml1 = dmin * (float)((sc1 >> 4) & 0x0F);
                #pragma unroll
                for (int l = 0; l < 16; ++l) {
                    int raw = (int)((qs[q_base + l] >> shift) & 0x3);
                    acc += xb[y_off + l] * (dl1 * (float)raw - ml1);
                }
                unsigned char sc2 = scales[is++];
                float dl2 = d * (float)(sc2 & 0x0F);
                float ml2 = dmin * (float)((sc2 >> 4) & 0x0F);
                #pragma unroll
                for (int l = 0; l < 16; ++l) {
                    int raw = (int)((qs[q_base + l + 16] >> shift) & 0x3);
                    acc += xb[y_off + 16 + l] * (dl2 * (float)raw - ml2);
                }
                y_off += 32;
                shift += 2;
            }
            q_base += 32;
        }
    }
    y[(long)m * N + n] = acc;
}

// linear with Q3_K packed weights. x:[M, K] f32; w:[N, K] packed Q3_K; y:[M, N] f32.
// Q3_K super-block (256 elements per block, 110 bytes):
//   bytes [0..32]    u8  hmask[32]    -- high 3rd bit per value, 1 bit per value
//   bytes [32..96]   u8  qs[64]       -- low 2 bits per value, 4 values per byte
//   bytes [96..108]  u8  scales[12]   -- 16 6-bit signed scales packed
//   bytes [108..110] f16 d            -- super-block scale
__global__ void linear_q3_k_f32(const float* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 float* __restrict__ y,
                                 int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 256;
    int row_bytes = n_blocks * 110;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp     = w_row + b * 110;
        const unsigned char* hmask  = bp + 0;
        const unsigned char* qs     = bp + 32;
        const unsigned char* scales = bp + 96;
        unsigned short d_bits = (unsigned short)bp[108] | ((unsigned short)bp[109] << 8);
        float d_all = f16_to_f32(d_bits);
        const float* xb = x_row + b * 256;

        // Unpack 16 signed 6-bit scales into local array.
        unsigned int aux0 = (unsigned int)scales[0] | ((unsigned int)scales[1] << 8) | ((unsigned int)scales[2] << 16) | ((unsigned int)scales[3] << 24);
        unsigned int aux1 = (unsigned int)scales[4] | ((unsigned int)scales[5] << 8) | ((unsigned int)scales[6] << 16) | ((unsigned int)scales[7] << 24);
        unsigned int aux2 = (unsigned int)scales[8] | ((unsigned int)scales[9] << 8) | ((unsigned int)scales[10] << 16) | ((unsigned int)scales[11] << 24);
        unsigned int tmp = aux2;
        unsigned int kmask1 = 0x03030303u;
        unsigned int kmask2 = 0x0f0f0f0fu;
        unsigned int new_aux3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        unsigned int new_aux2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        unsigned int new_aux1 = ( aux1       & kmask2) | (((tmp >> 2) & kmask1) << 4);
        unsigned int new_aux0 = ( aux0       & kmask2) | (((tmp >> 0) & kmask1) << 4);
        signed char sc[16];
        sc[0]  = (signed char)( new_aux0        & 0xFF);
        sc[1]  = (signed char)((new_aux0 >>  8) & 0xFF);
        sc[2]  = (signed char)((new_aux0 >> 16) & 0xFF);
        sc[3]  = (signed char)((new_aux0 >> 24) & 0xFF);
        sc[4]  = (signed char)( new_aux1        & 0xFF);
        sc[5]  = (signed char)((new_aux1 >>  8) & 0xFF);
        sc[6]  = (signed char)((new_aux1 >> 16) & 0xFF);
        sc[7]  = (signed char)((new_aux1 >> 24) & 0xFF);
        sc[8]  = (signed char)( new_aux2        & 0xFF);
        sc[9]  = (signed char)((new_aux2 >>  8) & 0xFF);
        sc[10] = (signed char)((new_aux2 >> 16) & 0xFF);
        sc[11] = (signed char)((new_aux2 >> 24) & 0xFF);
        sc[12] = (signed char)( new_aux3        & 0xFF);
        sc[13] = (signed char)((new_aux3 >>  8) & 0xFF);
        sc[14] = (signed char)((new_aux3 >> 16) & 0xFF);
        sc[15] = (signed char)((new_aux3 >> 24) & 0xFF);

        int y_off = 0;
        int q_base = 0;
        unsigned char m_bit = 1;
        int is = 0;
        for (int outer = 0; outer < 2; ++outer) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl1 = d_all * (float)((int)sc[is] - 32); is += 1;
                #pragma unroll
                for (int l = 0; l < 16; ++l) {
                    int raw = (int)((qs[q_base + l] >> shift) & 0x3);
                    int hm = (hmask[l] & m_bit) ? 0 : 4;
                    int q3 = raw - hm;
                    acc += xb[y_off + l] * dl1 * (float)q3;
                }
                float dl2 = d_all * (float)((int)sc[is] - 32); is += 1;
                #pragma unroll
                for (int l = 0; l < 16; ++l) {
                    int raw = (int)((qs[q_base + l + 16] >> shift) & 0x3);
                    int hm = (hmask[l + 16] & m_bit) ? 0 : 4;
                    int q3 = raw - hm;
                    acc += xb[y_off + 16 + l] * dl2 * (float)q3;
                }
                y_off += 32;
                shift += 2;
                m_bit <<= 1;
            }
            q_base += 32;
        }
    }
    y[(long)m * N + n] = acc;
}

// Cooperative warp GEMV for Q5_K, decode-time M=1 fast path. Same partitioning
// scheme as the Q4_K coop kernel (each thread handles K/1024 contiguous
// sub-blocks, warp-shuffle reduce) — Q5_K shares Q4_K's super-block geometry
// (256 elem / 8 sub-blocks of 32) but adds a 32-byte qh array of 5th high bits.
// Requires K % 1024 == 0 (true of any 4096/8192/12288-class hidden/ff dim).
__global__ void linear_q5_k_gemv_coop_f32(const float* __restrict__ x,
                                           const unsigned char* __restrict__ w,
                                           float* __restrict__ y,
                                           int N, int K) {
    int n = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= N) return;
    int t = threadIdx.x;          // [0, 32)
    int n_blocks    = K / 256;
    int total_sb    = n_blocks * 8;
    int sb_per_thd  = total_sb / 32;
    int sb_start    = t * sb_per_thd;
    int sb_end      = sb_start + sb_per_thd;
    int row_bytes   = n_blocks * 176;
    const unsigned char* w_row = w + (long)n * row_bytes;

    // Cache the most recent superblock header across consecutive sub-blocks.
    int last_b = -1;
    float d_sb = 0.0f, min_sb = 0.0f;
    const unsigned char* qh_sb = nullptr;
    const unsigned char* qs_sb = nullptr;
    const unsigned char* sc_sb = nullptr;

    float acc = 0.0f;
    for (int sb = sb_start; sb < sb_end; ++sb) {
        int b  = sb >> 3;             // / 8
        int is = sb & 7;              // % 8
        if (b != last_b) {
            const unsigned char* bp = w_row + b * 176;
            unsigned short d_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
            unsigned short m_bits = (unsigned short)bp[2] | ((unsigned short)bp[3] << 8);
            d_sb   = f16_to_f32(d_bits);
            min_sb = f16_to_f32(m_bits);
            sc_sb  = bp + 4;
            qh_sb  = bp + 16;
            qs_sb  = bp + 48;
            last_b = b;
        }
        unsigned char sc_byte, mm_byte;
        q4k_unpack_scale_min(is, sc_sb, &sc_byte, &mm_byte);
        float dq  = d_sb   * (float)sc_byte;
        float mq  = min_sb * (float)mm_byte;
        int qs_off = (is >> 1) * 32;
        int x_off  = b * 256 + is * 32;
        const unsigned char* qsp = qs_sb + qs_off;
        const float* xp = x + x_off;
        unsigned char hmask = (unsigned char)(1u << is);
        if ((is & 1) == 0) {
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                int high = (qh_sb[l] & hmask) ? 16 : 0;
                float qv = (float)((qsp[l] & 0x0F) + high);
                acc += xp[l] * (dq * qv - mq);
            }
        } else {
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                int high = (qh_sb[l] & hmask) ? 16 : 0;
                float qv = (float)(((qsp[l] >> 4) & 0x0F) + high);
                acc += xp[l] * (dq * qv - mq);
            }
        }
    }

    // Warp-reduce the partial sums into thread 0.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (t == 0) y[n] = acc;
}

// linear with Q5_K packed weights. Same scale/min packing as Q4_K, but each
// nibble gets a 5th high bit pulled from a separate 32-byte qh array.
// Q5_K super-block (256 elements per block, 176 bytes):
//   bytes [0..2]    f16 d
//   bytes [2..4]    f16 dmin
//   bytes [4..16]   u8  scales[12]   (8 6-bit scales + 8 6-bit mins, packed; same as Q4_K)
//   bytes [16..48]  u8  qh[32]       (5th bit of each value, 8 sub-blocks per byte)
//   bytes [48..176] u8  qs[128]      (256 4-bit nibbles)
__global__ void linear_q5_k_f32(const float* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 float* __restrict__ y,
                                 int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 256;
    int row_bytes = n_blocks * 176;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 176;
        unsigned short d_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        unsigned short m_bits = (unsigned short)bp[2] | ((unsigned short)bp[3] << 8);
        float d   = f16_to_f32(d_bits);
        float min = f16_to_f32(m_bits);
        const unsigned char* scales = bp + 4;
        const unsigned char* qh     = bp + 16;
        const unsigned char* qs     = bp + 48;
        const float* xb = x_row + b * 256;

        int q_off = 0;
        int y_off = 0;
        unsigned char u1 = 0x01;
        unsigned char u2 = 0x02;
        for (int is = 0; is < 8; is += 2) {
            unsigned char sc1, mm1, sc2, mm2;
            q4k_unpack_scale_min(is + 0, scales, &sc1, &mm1);
            q4k_unpack_scale_min(is + 1, scales, &sc2, &mm2);
            float d1 = d * (float)sc1; float min1 = min * (float)mm1;
            float d2 = d * (float)sc2; float min2 = min * (float)mm2;

            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                int high = (qh[l] & u1) ? 16 : 0;
                float qv = (float)((qs[q_off + l] & 0x0F) + high);
                acc += xb[y_off + l] * (d1 * qv - min1);
            }
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                int high = (qh[l] & u2) ? 16 : 0;
                float qv = (float)(((qs[q_off + l] >> 4) & 0x0F) + high);
                acc += xb[y_off + 32 + l] * (d2 * qv - min2);
            }
            q_off += 32;
            y_off += 64;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    y[(long)m * N + n] = acc;
}

// linear with IQ4_NL packed weights. 32-elem blocks, 18 bytes each.
// Non-linear 4-bit quantization with a 16-entry signed lookup table.
//   bytes [0..2]   f16 d
//   bytes [2..18]  u8  qs[16]
// y_i = d * KVALUES_IQ4NL[nibble]
__device__ const signed char KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};
__global__ void linear_iq4_nl_f32(const float* __restrict__ x,
                                   const unsigned char* __restrict__ w,
                                   float* __restrict__ y,
                                   int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 32;
    int row_bytes = n_blocks * 18;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 18;
        unsigned short scale_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        float d = f16_to_f32(scale_bits);
        const unsigned char* qs = bp + 2;
        const float* xb = x_row + b * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int lo = (int)(qs[i] & 0x0F);
            int hi = (int)((qs[i] >> 4) & 0x0F);
            acc += xb[i]      * (float)KVALUES_IQ4NL[lo] * d;
            acc += xb[i + 16] * (float)KVALUES_IQ4NL[hi] * d;
        }
    }
    y[(long)m * N + n] = acc;
}

// linear with IQ4_XS packed weights. 256-elem super-blocks, 136 bytes each.
//   bytes [0..2]    f16 d
//   bytes [2..4]    u16 scales_h    (high 2 bits of each of 8 sub-block 6-bit scales)
//   bytes [4..8]    u8  scales_l[4] (low 4 bits, two sub-blocks per byte)
//   bytes [8..136]  u8  qs[128]     (256 4-bit indices into KVALUES_IQ4NL)
// Per sub-block ib (0..8):
//   ls = (scales_l[ib/2] >> (4*(ib&1)) & 0xF) | (((scales_h >> (2*ib)) & 0x3) << 4)
//   dl = d * (ls - 32)
__global__ void linear_iq4_xs_f32(const float* __restrict__ x,
                                   const unsigned char* __restrict__ w,
                                   float* __restrict__ y,
                                   int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 256;
    int row_bytes = n_blocks * 136;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 136;
        unsigned short d_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        float d = f16_to_f32(d_bits);
        unsigned int scales_h = (unsigned int)bp[2] | ((unsigned int)bp[3] << 8);
        const unsigned char* scales_l = bp + 4;
        const unsigned char* qs = bp + 8;
        const float* xb = x_row + b * 256;

        int y_off = 0;
        int q_off = 0;
        for (int ib = 0; ib < 8; ++ib) {
            int lo4 = (int)((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F);
            int hi2 = (int)((scales_h >> (2 * ib)) & 0x03) << 4;
            int ls  = lo4 | hi2;
            float dl = d * (float)(ls - 32);
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                int lo = (int)(qs[q_off + j] & 0x0F);
                int hi = (int)((qs[q_off + j] >> 4) & 0x0F);
                acc += xb[y_off + j]      * dl * (float)KVALUES_IQ4NL[lo];
                acc += xb[y_off + 16 + j] * dl * (float)KVALUES_IQ4NL[hi];
            }
            y_off += 32;
            q_off += 16;
        }
    }
    y[(long)m * N + n] = acc;
}

// linear with Q4_0 packed weights. 32-elem blocks, 18 bytes each.
//   bytes [0..2]   f16 d
//   bytes [2..18]  u8  qs[16]     (32 nibbles; low nibble = elem i, high nibble = elem i+16)
// y_i = (nibble - 8) * d
__global__ void linear_q4_0_f32(const float* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 float* __restrict__ y,
                                 int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 32;
    int row_bytes = n_blocks * 18;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 18;
        unsigned short scale_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        float d = f16_to_f32(scale_bits);
        const unsigned char* qs = bp + 2;
        const float* xb = x_row + b * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int lo = (int)(qs[i] & 0x0F)        - 8;
            int hi = (int)((qs[i] >> 4) & 0x0F) - 8;
            acc += xb[i]      * (float)lo * d;
            acc += xb[i + 16] * (float)hi * d;
        }
    }
    y[(long)m * N + n] = acc;
}

// linear with Q4_1 packed weights. 32-elem blocks, 20 bytes each.
//   bytes [0..2]   f16 d
//   bytes [2..4]   f16 m
//   bytes [4..20]  u8  qs[16]
// y_i = nibble * d + m
__global__ void linear_q4_1_f32(const float* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 float* __restrict__ y,
                                 int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 32;
    int row_bytes = n_blocks * 20;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 20;
        unsigned short d_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        unsigned short m_bits = (unsigned short)bp[2] | ((unsigned short)bp[3] << 8);
        float d = f16_to_f32(d_bits);
        float mm = f16_to_f32(m_bits);
        const unsigned char* qs = bp + 4;
        const float* xb = x_row + b * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int lo = (int)(qs[i] & 0x0F);
            int hi = (int)((qs[i] >> 4) & 0x0F);
            acc += xb[i]      * ((float)lo * d + mm);
            acc += xb[i + 16] * ((float)hi * d + mm);
        }
    }
    y[(long)m * N + n] = acc;
}

// Cooperative warp GEMV for Q5_0, decode-time M=1 fast path. Same
// element-parallel design as the Q8_0 coop: each thread handles ONE element
// per block. Q5_0 packs the block's 32 elements as 16 (lo, hi) pairs in the
// 16-byte qs[]; threads 0..15 take the low nibble of qs[t], threads 16..31
// take the high nibble of qs[t-16] (so adjacent thread pairs (t, t+16) share
// a qs byte read). Works for any K%32==0 — unblocks Gemma 3-1B's K=1152
// attn_q/k + ffn_gate/up matmuls that don't fit any K%1024==0 kernel.
__global__ void linear_q5_0_gemv_coop_f32(const float* __restrict__ x,
                                           const unsigned char* __restrict__ w,
                                           float* __restrict__ y,
                                           int N, int K) {
    int n = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= N) return;
    int t = threadIdx.x;          // [0, 32)
    int idx = t < 16 ? t : t - 16;       // qs[] byte index
    int hi  = t >= 16 ? 1 : 0;           // 0 = low nibble, 1 = high nibble
    int n_blocks  = K / 32;
    int row_bytes = n_blocks * 22;
    const unsigned char* w_row = w + (long)n * row_bytes;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 22;
        unsigned short scale_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        float d = f16_to_f32(scale_bits);
        unsigned int qh = (unsigned int)bp[2]
                        | ((unsigned int)bp[3] << 8)
                        | ((unsigned int)bp[4] << 16)
                        | ((unsigned int)bp[5] << 24);
        const unsigned char* qs = bp + 6;
        const float* xb = x + b * 32;

        unsigned char qb = qs[idx];
        int nibble = hi ? ((qb >> 4) & 0x0F) : (qb & 0x0F);
        int qh_bit = (int)((qh >> (idx + (hi ? 16 : 0))) & 0x1u);
        int v = (nibble | (qh_bit << 4)) - 16;
        acc += xb[t] * (float)v * d;
    }

    // Warp-reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (t == 0) y[n] = acc;
}

// linear with Q5_0 packed weights. 32-elem blocks, 22 bytes each.
//   bytes [0..2]   f16 d
//   bytes [2..6]   u32 qh         (5th bit of each value)
//   bytes [6..22]  u8  qs[16]     (32 nibbles)
// y_i = ((nibble | (qh_bit << 4)) - 16) * d
__global__ void linear_q5_0_f32(const float* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 float* __restrict__ y,
                                 int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 32;
    int row_bytes = n_blocks * 22;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 22;
        unsigned short scale_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        float d = f16_to_f32(scale_bits);
        unsigned int qh = (unsigned int)bp[2]
                        | ((unsigned int)bp[3] << 8)
                        | ((unsigned int)bp[4] << 16)
                        | ((unsigned int)bp[5] << 24);
        const unsigned char* qs = bp + 6;
        const float* xb = x_row + b * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int xh0 = (int)((qh >> i) & 0x1u);
            int xh1 = (int)((qh >> (i + 16)) & 0x1u);
            int lo  = (int)(qs[i] & 0x0F)        | (xh0 << 4);
            int hi  = (int)((qs[i] >> 4) & 0x0F) | (xh1 << 4);
            acc += xb[i]      * (float)(lo - 16) * d;
            acc += xb[i + 16] * (float)(hi - 16) * d;
        }
    }
    y[(long)m * N + n] = acc;
}

// Cooperative warp GEMV for Q5_1, decode-time M=1 fast path. Same scheme as
// the Q5_0 coop kernel: each thread handles one element per block. Q5_1's only
// delta is a 2-byte `m` field after `d`, and the per-element formula is
// `nibble * d + m` (no -16 offset). Works for any K%32==0 — Gemma 3n e2b's
// matmuls are 50% Q5_1 with K=2048 (which IS %1024-aligned so a K-quants-style
// kernel could also work, but the element-parallel design is simpler and
// has equivalent coalescing on Q5_1).
__global__ void linear_q5_1_gemv_coop_f32(const float* __restrict__ x,
                                           const unsigned char* __restrict__ w,
                                           float* __restrict__ y,
                                           int N, int K) {
    int n = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= N) return;
    int t = threadIdx.x;          // [0, 32)
    int idx = t < 16 ? t : t - 16;
    int hi  = t >= 16 ? 1 : 0;
    int n_blocks  = K / 32;
    int row_bytes = n_blocks * 24;
    const unsigned char* w_row = w + (long)n * row_bytes;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 24;
        unsigned short d_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        unsigned short m_bits = (unsigned short)bp[2] | ((unsigned short)bp[3] << 8);
        float d  = f16_to_f32(d_bits);
        float mm = f16_to_f32(m_bits);
        unsigned int qh = (unsigned int)bp[4]
                        | ((unsigned int)bp[5] << 8)
                        | ((unsigned int)bp[6] << 16)
                        | ((unsigned int)bp[7] << 24);
        const unsigned char* qs = bp + 8;
        const float* xb = x + b * 32;

        unsigned char qb = qs[idx];
        int nibble = hi ? ((qb >> 4) & 0x0F) : (qb & 0x0F);
        int qh_bit = (int)((qh >> (idx + (hi ? 16 : 0))) & 0x1u);
        int v = nibble | (qh_bit << 4);
        acc += xb[t] * ((float)v * d + mm);
    }

    // Warp-reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (t == 0) y[n] = acc;
}

// linear with Q5_1 packed weights. 32-elem blocks, 24 bytes each.
//   bytes [0..2]   f16 d
//   bytes [2..4]   f16 m
//   bytes [4..8]   u32 qh         (5th bit of each value)
//   bytes [8..24]  u8  qs[16]     (32 nibbles)
// y_i = (nibble | (qh_bit << 4)) * d + m
__global__ void linear_q5_1_f32(const float* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 float* __restrict__ y,
                                 int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    int n_blocks = K / 32;
    int row_bytes = n_blocks * 24;
    const unsigned char* w_row = w + (long)n * row_bytes;
    const float* x_row = x + (long)m * K;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 24;
        unsigned short d_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        unsigned short m_bits = (unsigned short)bp[2] | ((unsigned short)bp[3] << 8);
        float d = f16_to_f32(d_bits);
        float mm = f16_to_f32(m_bits);
        unsigned int qh = (unsigned int)bp[4]
                        | ((unsigned int)bp[5] << 8)
                        | ((unsigned int)bp[6] << 16)
                        | ((unsigned int)bp[7] << 24);
        const unsigned char* qs = bp + 8;
        const float* xb = x_row + b * 32;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int xh0 = (int)((qh >> i) & 0x1u);
            int xh1 = (int)((qh >> (i + 16)) & 0x1u);
            int lo  = (int)(qs[i] & 0x0F)        | (xh0 << 4);
            int hi  = (int)((qs[i] >> 4) & 0x0F) | (xh1 << 4);
            acc += xb[i]      * ((float)lo * d + mm);
            acc += xb[i + 16] * ((float)hi * d + mm);
        }
    }
    y[(long)m * N + n] = acc;
}

// linear: c[m, n] = sum_k a[m, k] * b[n, k]
// a: [M, K] row-major; b: [N, K] row-major; c: [M, N] row-major.
// One thread per output element. Block 16x16.
__global__ void linear_f32(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c,
                            int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    const float* a_row = a + m * K;
    const float* b_row = b + n * K;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += a_row[k] * b_row[k];
    }
    c[m * N + n] = acc;
}

// rmsnorm: y[r, i] = x[r, i] / sqrt(mean(x[r]^2) + eps) * w[i]
// One block per row; threads cooperate on the reduction.
__global__ void rmsnorm_f32(const float* __restrict__ x,
                             const float* __restrict__ w,
                             float* __restrict__ y,
                             int n_rows, int last, float eps) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs = blockDim.x;

    float local = 0.0f;
    for (int i = tid; i < last; i += bs) {
        float v = x[row * last + i];
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();

    // Tree reduction within block. Assumes bs is a power of 2.
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)last + eps);

    for (int i = tid; i < last; i += bs) {
        y[row * last + i] = x[row * last + i] * inv_rms * w[i];
    }
}

// Fused residual add + rmsnorm: x += y; out = rmsnorm(x, w).
// Replaces the `add_inplace(x, y)` + `rmsnorm(x, w)` pair every transformer
// block does between attention/FFN sublayers. Saves one launch and one full
// read of `y` (the residual is consumed exactly once instead of being added
// to x in pass A and then x re-read in pass B).
__global__ void add_inplace_then_rmsnorm_f32(float* __restrict__ x,
                                              const float* __restrict__ y,
                                              const float* __restrict__ w,
                                              float* __restrict__ out,
                                              int n_rows, int last, float eps) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs = blockDim.x;

    // Pass 1: write back x[i] += y[i] and accumulate (x_new)^2 for rmsnorm.
    float local = 0.0f;
    for (int i = tid; i < last; i += bs) {
        int off = row * last + i;
        float v = x[off] + y[off];
        x[off] = v;
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)last + eps);

    // Pass 2: out[i] = x_new[i] * inv_rms * w[i].
    for (int i = tid; i < last; i += bs) {
        int off = row * last + i;
        out[off] = x[off] * inv_rms * w[i];
    }
}

// Full LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b.
// Two reductions sharing the same shmem buffer: pass 1 sums for mean, then
// (after using the mean) pass 2 sums squared deviations for var. Used by
// SigLIP/CLIP/Qwen3-VL vision towers.
__global__ void layer_norm_f32(const float* __restrict__ x,
                                const float* __restrict__ w,
                                const float* __restrict__ b,
                                float* __restrict__ y,
                                int n_rows, int last, float eps) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    // ----- Pass 1: row sum → mean -----
    float local = 0.0f;
    for (int i = tid; i < last; i += bs) {
        local += x[row * last + i];
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)last;
    __syncthreads();

    // ----- Pass 2: sum (x - mean)^2 → var -----
    local = 0.0f;
    for (int i = tid; i < last; i += bs) {
        float v = x[row * last + i] - mean;
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sdata[0] / (float)last + eps);

    // ----- Apply: (x - mean) * inv * w + b -----
    for (int i = tid; i < last; i += bs) {
        y[row * last + i] = (x[row * last + i] - mean) * inv * w[i] + b[i];
    }
}

// Bias add along the last axis: x[r, j] += bias[j], in-place. Cheap per-thread
// kernel — one thread per (row, lane) pair.
__global__ void add_bias_last_f32(float* __restrict__ x,
                                   const float* __restrict__ bias,
                                   int n_rows, int last) {
    int row  = blockIdx.x;
    int lane = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= n_rows || lane >= last) return;
    x[row * last + lane] += bias[lane];
}

// rmsnorm without learnable scale: y[r, i] = x[r, i] / sqrt(mean(x[r]^2) + eps).
// Used by Gemma 3n V-norm. Same shape as rmsnorm_f32 minus the per-element weight mul.
__global__ void rmsnorm_no_scale_f32(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int n_rows, int last, float eps) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs = blockDim.x;

    float local = 0.0f;
    for (int i = tid; i < last; i += bs) {
        float v = x[row * last + i];
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)last + eps);

    for (int i = tid; i < last; i += bs) {
        y[row * last + i] = x[row * last + i] * inv_rms;
    }
}

// In-place softmax along the last axis.
__global__ void softmax_last_f32(float* __restrict__ x,
                                  int n_rows, int last) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs = blockDim.x;

    // Pass 1: max.
    float local_max = -INFINITY;
    for (int i = tid; i < last; i += bs) {
        float v = x[row * last + i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float m = sdata[0];
    __syncthreads();

    // Pass 2: exp(x - max), accumulate sum.
    float local_sum = 0.0f;
    for (int i = tid; i < last; i += bs) {
        float e = expf(x[row * last + i] - m);
        x[row * last + i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float total = sdata[0];
    if (total <= 0.0f) total = 1.0f;  // defensive — should never happen
    float inv = 1.0f / total;

    for (int i = tid; i < last; i += bs) {
        x[row * last + i] *= inv;
    }
}

__global__ void silu_f32(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = v / (1.0f + expf(-v));
}

// Fused silu(a) * b. Used by SwiGLU FFN.
__global__ void silu_mul_f32(const float* __restrict__ a, const float* __restrict__ b,
                              float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i];
    y[i] = (v / (1.0f + expf(-v))) * b[i];
}

__global__ void gelu_approx_f32(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float COEFF = 0.044715f;
    float v = x[i];
    float inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
    y[i] = 0.5f * v * (1.0f + tanhf(inner));
}

// Fused GeLU(a) * b. Mirrors `silu_mul_f32`. Used by Gemma 4 MoE's gated MLP.
__global__ void gelu_approx_mul_f32(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float COEFF = 0.044715f;
    float v = a[i];
    float inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
    float gelu_v = 0.5f * v * (1.0f + tanhf(inner));
    y[i] = gelu_v * b[i];
}

// silu_mul_split: out[s, j] = silu(fused[s, j]) * fused[s, ff + j].
// `fused` is [seq, 2*ff] (output of one matmul against stacked [gate; up]
// weights). Replaces split + silu_mul. 2D grid: ff along x, seq along y.
// VENDORED-LOCAL: GLM-5.3-Flash clamped SwiGLU. after_silu != 0 clamps the
// activation (the text FFN); == 0 clamps the pre-activation (the vision tower).
// limit <= 0 disables the clamp.
__global__ void swiglu_clamped_f32(const float* __restrict__ gate,
                                   const float* __restrict__ up,
                                   float* __restrict__ y,
                                   int n, float limit, int after_silu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = gate[i];
    float b = up[i];
    const bool clamped = limit > 0.0f;
    if (clamped && after_silu == 0 && a > limit) a = limit;
    a = a / (1.0f + expf(-a));
    if (clamped && after_silu != 0 && a > limit) a = limit;
    if (clamped) b = fminf(fmaxf(b, -limit), limit);
    y[i] = a * b;
}

// VENDORED-LOCAL: GLM-5.3-Flash clamped SwiGLU over a fused gate|up row.
__global__ void swiglu_clamped_split_f32(const float* __restrict__ fused,
                                         float* __restrict__ out,
                                         int seq, int ff,
                                         float limit, int after_silu) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (j >= ff || s >= seq) return;
    int in_off = s * 2 * ff;
    float a = fused[in_off + j];
    float b = fused[in_off + ff + j];
    const bool clamped = limit > 0.0f;
    if (clamped && after_silu == 0 && a > limit) a = limit;
    a = a / (1.0f + __expf(-a));
    if (clamped && after_silu != 0 && a > limit) a = limit;
    if (clamped) b = fminf(fmaxf(b, -limit), limit);
    out[s * ff + j] = a * b;
}

// VENDORED-LOCAL: GLM-5.3-Flash. One warp per output row of a batched GEMV:
// w is [B, M, K] row-major, x is [B, K], y is [B, M]. Lanes stride K by 32 so
// the reads of w coalesce, then reduce in the warp.
__global__ void batched_gemv_f32(const float* __restrict__ w,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int B, int M, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int b = blockIdx.y;
    const int m = blockIdx.x * warps + warp;
    if (b >= B || m >= M) return;
    const float* wrow = w + ((size_t)b * M + m) * (size_t)K;
    const float* xrow = x + (size_t)b * (size_t)K;
    float acc = 0.0f;
    for (int i = lane; i < K; i += 32) acc += wrow[i] * xrow[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[(size_t)b * (size_t)M + m] = acc;
}

// VENDORED-LOCAL: GLM-5.3-Flash. One KDA delta-rule step, one warp per state
// row. state is [H, D, D], qkvg is [4, H*D] (q, k, v, g_log), beta is [H],
// out is [H*D]. Rows are independent, so there is no scan here at all.
__global__ void kda_delta_step_f32(float* __restrict__ state,
                                   const float* __restrict__ qkvg,
                                   const float* __restrict__ beta,
                                   float* __restrict__ out,
                                   int n_head, int hd, float scale) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int row = blockIdx.x * warps + warp;
    const int n = n_head * hd;
    if (row >= n) return;
    const int h = row / hd;

    float* st = state + (size_t)row * (size_t)hd;
    const float* q = qkvg;
    const float* k = qkvg + (size_t)n;
    const float* v = qkvg + (size_t)2 * n;
    const float* g = qkvg + (size_t)3 * n;
    const float* kh = k + (size_t)h * hd;
    const float* qh = q + (size_t)h * hd;

    const float decay = __expf(g[row]);
    // pass 1: decay the row in place, then dot it with k
    float acc = 0.0f;
    for (int j = lane; j < hd; j += 32) {
        float r = st[j] * decay;
        st[j] = r;
        acc += r * kh[j];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    acc = __shfl_sync(0xffffffff, acc, 0);
    // the delta rule: write only the part of v the state does not predict
    const float d = (v[row] - acc) * beta[h];

    // pass 2: rank-1 update, then dot with q
    float o2 = 0.0f;
    for (int j = lane; j < hd; j += 32) {
        float r = st[j] + d * kh[j];
        st[j] = r;
        o2 += r * qh[j];
    }
    for (int o = 16; o > 0; o >>= 1) o2 += __shfl_down_sync(0xffffffff, o2, o);
    if (lane == 0) out[row] = o2 * scale;
}

__global__ void silu_mul_split_f32(const float* __restrict__ fused,
                                    float* __restrict__ out,
                                    int seq, int ff) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (j >= ff || s >= seq) return;
    int two_ff = 2 * ff;
    int in_off  = s * two_ff;
    float g = fused[in_off + j];
    float u = fused[in_off + ff + j];
    out[s * ff + j] = (g / (1.0f + __expf(-g))) * u;
}

// Element-wise sigmoid: y[i] = 1 / (1 + exp(-x[i])). Used by Qwen3.5/3.6
// gated attention output and qwen35moe shared-expert gate.
__global__ void sigmoid_f32(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = 1.0f / (1.0f + __expf(-x[i]));
}

// Fused in-place: x[i] *= sigmoid(gate[i]). Replaces sigmoid + mul_inplace
// (saves one launch + a global write/read of the sigmoid intermediate).
__global__ void mul_sigmoid_inplace_f32(float* __restrict__ x,
                                          const float* __restrict__ gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 1.0f / (1.0f + __expf(-gate[i]));
    x[i] *= s;
}

// gelu_approx_mul_split: out[s, j] = gelu_approx(fused[s, j]) * fused[s, ff+j].
// GeGLU equivalent of `silu_mul_split_f32`. Used by Gemma family.
__global__ void gelu_approx_mul_split_f32(const float* __restrict__ fused,
                                            float* __restrict__ out,
                                            int seq, int ff) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (j >= ff || s >= seq) return;
    int two_ff = 2 * ff;
    int in_off  = s * two_ff;
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float COEFF = 0.044715f;
    float g = fused[in_off + j];
    float u = fused[in_off + ff + j];
    float inner = SQRT_2_OVER_PI * (g + COEFF * g * g * g);
    float gelu_v = 0.5f * g * (1.0f + tanhf(inner));
    out[s * ff + j] = gelu_v * u;
}

// Element-wise tanh in place.
__global__ void tanh_inplace_f32(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}

// Gaussian-top-k mask in place (Gemma 3n activation sparsity).
// Per row: cutoff = mean + std_multiplier * sqrt(var); x[r, j] = relu(x[r, j] - cutoff).
// One block per row; two reduction passes (sum, then sum of squared deviations).
// Shared memory holds blockDim.x floats for the reduction.
__global__ void gaussian_topk_inplace_f32(float* __restrict__ x,
                                           int n_rows, int last,
                                           float std_multiplier) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs = blockDim.x;

    // Pass 1: sum → mean.
    float local = 0.0f;
    for (int i = tid; i < last; i += bs) local += x[row * last + i];
    sdata[tid] = local;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)last;
    __syncthreads();

    // Pass 2: sum of squared deviations → variance → std.
    float local_v = 0.0f;
    for (int i = tid; i < last; i += bs) {
        float d = x[row * last + i] - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float std = sqrtf(sdata[0] / (float)last);
    float cutoff = mean + std * std_multiplier;

    for (int i = tid; i < last; i += bs) {
        float v = x[row * last + i] - cutoff;
        x[row * last + i] = v > 0.0f ? v : 0.0f;
    }
}

// AltUp predict (Gemma 3n). For each output stream i:
//   predicted[i, s, h] = streams[i, s, h] + sum_j coefs[s, i*n_alt + j] * streams[j, s, h]
// streams + predicted shape: [n_alt, seq, hidden]
// coefs shape: [seq, n_alt * n_alt] (raw output of linear, no transpose).
__global__ void altup_predict_f32(
    const float* __restrict__ streams,
    const float* __restrict__ coefs,
    float* __restrict__ predicted,
    int n_alt, int seq, int hidden
) {
    int s = blockIdx.z;
    int i = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seq || i >= n_alt || h >= hidden) return;

    int out_off = (i * seq + s) * hidden + h;
    int coef_off = s * n_alt * n_alt + i * n_alt;

    // Residual: + stream[i].
    float sum = streams[out_off];
    for (int j = 0; j < n_alt; j++) {
        int in_off = (j * seq + s) * hidden + h;
        sum += coefs[coef_off + j] * streams[in_off];
    }
    predicted[out_off] = sum;
}

// AltUp correct (Gemma 3n). For each stream i:
//   innovation = activated[s, h] - predictions[active_idx, s, h]
//   corrected[i, s, h] = predictions[i, s, h] + (coefs[s, i] + 1.0) * innovation
__global__ void altup_correct_f32(
    const float* __restrict__ predictions,
    const float* __restrict__ activated,
    const float* __restrict__ coefs,
    float* __restrict__ corrected,
    int n_alt, int seq, int hidden, int active_idx
) {
    int s = blockIdx.z;
    int i = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seq || i >= n_alt || h >= hidden) return;

    int act_off       = s * hidden + h;
    int pred_active   = (active_idx * seq + s) * hidden + h;
    int pred_i        = (i * seq + s) * hidden + h;

    float innovation  = activated[act_off] - predictions[pred_active];
    float coef_plus_1 = coefs[s * n_alt + i] + 1.0f;
    corrected[pred_i] = predictions[pred_i] + coef_plus_1 * innovation;
}

__global__ void add_inplace_f32(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void mul_inplace_f32(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= y[i];
}

__global__ void mul_scalar_inplace_f32(float* __restrict__ x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// Broadcast multiply along the last axis: x[..., j] *= w[j].
// One thread per output element; 2D grid (last-axis along x, row index along y).
__global__ void mul_inplace_broadcast_last_f32(float* __restrict__ x,
                                                const float* __restrict__ w,
                                                int n_rows, int last) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (j >= last || r >= n_rows) return;
    x[r * last + j] *= w[j];
}

// Per-row scalar gate: x[s, j] *= g[s]. `g` is contiguous [seq] (a [seq,1]
// view aliases the same buffer). 2D grid: inner along x, seq along y.
__global__ void mul_inplace_broadcast_axis0_f32(float* __restrict__ x,
                                                  const float* __restrict__ g,
                                                  int seq, int inner) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (j >= inner || s >= seq) return;
    x[s * inner + j] *= g[s];
}

// Slice one axis-1 idx out of a 3D tensor: out[s, j] = src[s, idx, j].
// src shape [d0, d1, d2]; out shape [d0, d2]. 2D grid (d2 along x, d0 along y).
__global__ void slice_axis1_2d_f32(const float* __restrict__ src,
                                    float* __restrict__ out,
                                    int d0, int d1, int d2, int idx) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (j >= d2 || s >= d0) return;
    int in_off  = (s * d1 + idx) * d2 + j;
    int out_off = s * d2 + j;
    out[out_off] = src[in_off];
}

// Broadcast add: dst[start..start+count, ...] += src.
// dst flat offset for slice s, inner index j is (start + s) * inner + j.
// Threads cover the (count * inner) work with a 2D grid (inner along x, count along y).
__global__ void add_to_axis0_range_f32(float* __restrict__ dst,
                                        const float* __restrict__ src,
                                        int start, int count, int inner) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (j >= inner || s >= count) return;
    int off = (start + s) * inner + j;
    dst[off] += src[j];
}

// Fused mul + axis0-range add: dst[start..start+count, ...] += src * scale.
// Replaces a `mul_scalar_inplace(src, scale)` + `add_to_axis0_range(dst, start, count, src)`
// pair. Saves one launch + one extra global write/read of `src` per call. Used by
// MoE accumulation (per-(token, expert) routing weight folded into the add).
__global__ void add_to_axis0_range_scaled_f32(float* __restrict__ dst,
                                                const float* __restrict__ src,
                                                int start, int count, int inner,
                                                float scale) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (j >= inner || s >= count) return;
    int off = (start + s) * inner + j;
    dst[off] += src[j] * scale;
}

// rope (in-place). x: [seq, n_heads, head_dim]
//   rope_type: 0 = Normal (interleaved (2k, 2k+1)),
//              1 = NeoX   (rotated half (k, k+head_dim/2)).
//   freq_factors: nullptr or pointer to head_dim/2 floats. When non-null, divides
//     each per-frequency angle (long-rope / YaRN-style scaling).
__global__ void rope_f32(float* __restrict__ x,
                          const unsigned int* __restrict__ positions,
                          int seq, int n_heads, int head_dim,
                          int rope_type, float theta,
                          const float* __restrict__ freq_factors) {
    int s = blockIdx.z;
    int h = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    if (s >= seq || h >= n_heads || k >= half) return;

    float pos = (float)positions[s];
    float freq = powf(theta, -2.0f * (float)k / (float)head_dim);
    float factor = (freq_factors != nullptr) ? freq_factors[k] : 1.0f;
    float angle = pos * freq / factor;
    float sin_v, cos_v;
    sincosf(angle, &sin_v, &cos_v);

    int off = (s * n_heads + h) * head_dim;
    int i_a, i_b;
    if (rope_type == 0) {
        i_a = off + 2 * k;
        i_b = off + 2 * k + 1;
    } else {
        i_a = off + k;
        i_b = off + k + half;
    }
    float a = x[i_a];
    float b = x[i_b];
    x[i_a] = a * cos_v - b * sin_v;
    x[i_b] = a * sin_v + b * cos_v;
}

// Partial NeoX RoPE: rotate only the FIRST rotated_dim dims of each head's
// head_dim-wide slice (Qwen3.5 partial_rotary_factor=0.25 ⇒ rotated_dim=64 of
// 256 head_dim). Pair = (off + k, off + k + rotated_dim/2) for k in [0, half).
// Dims [rotated_dim..head_dim) untouched.
__global__ void rope_partial_neox_f32(float* __restrict__ x,
                                       const unsigned int* __restrict__ positions,
                                       int seq, int n_heads, int head_dim,
                                       int rotated_dim, float theta) {
    int s = blockIdx.z;
    int h = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int half = rotated_dim / 2;
    if (s >= seq || h >= n_heads || k >= half) return;

    float pos = (float)positions[s];
    float freq = powf(theta, -2.0f * (float)k / (float)rotated_dim);
    float angle = pos * freq;
    float sin_v, cos_v;
    sincosf(angle, &sin_v, &cos_v);

    int off = (s * n_heads + h) * head_dim;
    int i_a = off + k;
    int i_b = off + k + half;
    float a = x[i_a];
    float b = x[i_b];
    x[i_a] = a * cos_v - b * sin_v;
    x[i_b] = a * sin_v + b * cos_v;
}

// repeat_kv: tile heads. x: [seq, n_kv, head_dim] -> y: [seq, n_kv*n_rep, head_dim].
__global__ void repeat_kv_f32(const float* __restrict__ x, float* __restrict__ y,
                               int seq, int n_kv, int n_rep, int head_dim) {
    int s   = blockIdx.z;
    int idx = blockIdx.y;                         // 0..n_kv*n_rep
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seq || idx >= n_kv * n_rep || d >= head_dim) return;
    int k = idx / n_rep;                          // source kv-head
    int src = (s * n_kv + k) * head_dim + d;
    int dst = (s * n_kv * n_rep + idx) * head_dim + d;
    y[dst] = x[src];
}

// Batched per-head Q·K^T with causal mask.
// q:      [seq,    n_h, hd]
// k:      [kv_len, n_h, hd]
// scores: [seq,    n_h, kv_len]
// One thread per (s, h, t).
__global__ void bmm_qkt_f32(const float* __restrict__ q,
                             const float* __restrict__ k,
                             float* __restrict__ scores,
                             int seq, int n_h, int kv_len, int hd,
                             float scale, int past) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z;
    if (t >= kv_len || h >= n_h || s >= seq) return;

    int max_t = past + s;
    float out;
    if (t > max_t) {
        out = -INFINITY;
    } else {
        const float* q_row = q + (s * n_h + h) * hd;
        const float* k_row = k + (t * n_h + h) * hd;
        float acc = 0.0f;
        for (int d = 0; d < hd; ++d) {
            acc += q_row[d] * k_row[d];
        }
        out = acc * scale;
    }
    scores[(s * n_h + h) * kv_len + t] = out;
}

// Batched per-head scores·V.
// scores: [seq, n_h, kv_len]
// v:      [kv_len, n_h, hd]
// out:    [seq, n_h, hd]
// One thread per (s, h, d).
__global__ void bmm_av_f32(const float* __restrict__ scores,
                            const float* __restrict__ v,
                            float* __restrict__ out,
                            int seq, int n_h, int kv_len, int hd) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z;
    if (d >= hd || h >= n_h || s >= seq) return;

    const float* sc_row = scores + (s * n_h + h) * kv_len;
    float acc = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
        acc += sc_row[t] * v[(t * n_h + h) * hd + d];
    }
    out[(s * n_h + h) * hd + d] = acc;
}

// Fused attention with GQA support and KV-cache prefix. Replaces
// (slice_axis0 + repeat_kv + bmm_qkt + softmax + bmm_av) with one kernel.
//
//   q:           [seq, n_h_q, hd]
//   k_buffer:    [max_kv_len, n_h_kv, hd]   (only [0..kv_len) is read)
//   v_buffer:    [max_kv_len, n_h_kv, hd]
//   out:         [seq, n_h_q, hd]
//
// `n_rep = n_h_q / n_h_kv`. Each Q head h reads KV head h / n_rep.
// `past` controls the causal mask: position t > past + s is masked.
//
// Layout: one block per (s, h_q). Shared memory holds the row of scores
// plus a small reduction buffer.
__global__ void attention_f32(const float* __restrict__ q,
                               const float* __restrict__ k_buf,
                               const float* __restrict__ v_buf,
                               float* __restrict__ out,
                               int seq, int n_h_q, int n_h_kv,
                               int max_kv_len, int kv_len, int hd,
                               float scale, int past, int sliding_window) {
    int h_q = blockIdx.x;
    int s = blockIdx.y;
    if (s >= seq || h_q >= n_h_q) return;

    int n_rep = n_h_q / n_h_kv;
    int h_kv = h_q / n_rep;

    extern __shared__ float smem[];
    float* scores = smem;
    float* reduce = smem + kv_len;

    int tid = threadIdx.x;
    int bs  = blockDim.x;
    int max_t = past + s;
    // Sliding-window lower bound: queries at position max_t attend to KV [min_t..max_t].
    // sliding_window <= 0 disables (full causal attention).
    int min_t = (sliding_window > 0 && max_t >= sliding_window) ? (max_t - sliding_window + 1) : 0;

    const float* q_row = q + (s * n_h_q + h_q) * hd;

    // Phase 1: scores[t] = Q · K[t, h_kv] * scale, with causal + (optional) sliding-window mask.
    for (int t = tid; t < kv_len; t += bs) {
        if (t > max_t || t < min_t) {
            scores[t] = -INFINITY;
            continue;
        }
        const float* k_row = k_buf + (t * n_h_kv + h_kv) * hd;
        float acc = 0.0f;
        for (int d = 0; d < hd; ++d) {
            acc += q_row[d] * k_row[d];
        }
        scores[t] = acc * scale;
    }
    __syncthreads();

    // Phase 2: max reduction.
    float local_max = -INFINITY;
    for (int t = tid; t < kv_len; t += bs) {
        if (scores[t] > local_max) local_max = scores[t];
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int r = bs / 2; r > 0; r >>= 1) {
        if (tid < r) reduce[tid] = fmaxf(reduce[tid], reduce[tid + r]);
        __syncthreads();
    }
    float maxv = reduce[0];
    __syncthreads();

    // Phase 3: exp and sum reduction.
    float local_sum = 0.0f;
    for (int t = tid; t < kv_len; t += bs) {
        float e = expf(scores[t] - maxv);
        scores[t] = e;
        local_sum += e;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int r = bs / 2; r > 0; r >>= 1) {
        if (tid < r) reduce[tid] += reduce[tid + r];
        __syncthreads();
    }
    float total = reduce[0];
    if (total <= 0.0f) total = 1.0f;
    float inv = 1.0f / total;
    __syncthreads();

    // Phase 4: out[s, h_q, d] = Σ_t (scores[t] * inv) * V[t, h_kv, d].
    // Note: max_kv_len isn't directly used here since we stride V by n_h_kv * hd
    // — V is contiguous [max_kv_len, n_h_kv, hd] and we only read t < kv_len.
    (void)max_kv_len;
    for (int d = tid; d < hd; d += bs) {
        float acc = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            acc += scores[t] * v_buf[(t * n_h_kv + h_kv) * hd + d];
        }
        out[(s * n_h_q + h_q) * hd + d] = acc * inv;
    }
}

// Embedding lookup. table: [V, D] row-major, tokens: [N], y: [N, D] row-major.
// One thread per output element.
__global__ void embed_lookup_f32(const float* __restrict__ table,
                                  const unsigned int* __restrict__ tokens,
                                  float* __restrict__ y,
                                  int n_tokens, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * d;
    if (i >= total) return;
    int tok = i / d;
    int col = i % d;
    int t = (int)tokens[tok];
    y[i] = table[t * d + col];
}

// argmax along last axis. Returns indices as u32 in `out`.
// Block per row; uses 2 * blockDim.x floats of shared mem (values + indices).
__global__ void argmax_last_f32(const float* __restrict__ x, unsigned int* out,
                                 int n_rows, int last) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float sdata[];
    int* sidx = (int*)(sdata + blockDim.x);

    int tid = threadIdx.x;
    int bs = blockDim.x;

    float local_v = -INFINITY;
    int   local_i = 0;
    for (int i = tid; i < last; i += bs) {
        float v = x[row * last + i];
        if (v > local_v) { local_v = v; local_i = i; }
    }
    sdata[tid] = local_v;
    sidx[tid]  = local_i;
    __syncthreads();

    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Earliest-index tie-break: the CPU argmax (and the greedy
            // sampler) keeps the FIRST maximal element; do the same so
            // device and host sampling agree on exact-f32 ties (SAMPLE-01).
            if (sdata[tid + s] > sdata[tid] ||
                (sdata[tid + s] == sdata[tid] && sidx[tid + s] < sidx[tid])) {
                sdata[tid] = sdata[tid + s];
                sidx[tid]  = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[row] = (unsigned int)sidx[0];
}

// Qwen3.5 attention layer Q+gate split: the joint q_proj output is laid out
// per-head as [query (head_dim) | gate (head_dim)]. This kernel splits along
// the per-head 2*head_dim block in one launch — one element per thread, no
// host roundtrip. Source `q_full` shape: [seq, n_heads, 2*head_dim] flat;
// destinations are [seq, n_heads, head_dim] each.
// Split a fused [seq, 3*d] QKV into three [seq, d] tensors. Each thread copies
// one (s, d_idx) element to all three outputs (with the right source offset).
// Used by the Qwen3-VL ViT block whose attn_qkv is a single [3*D, D] matmul.
__global__ void split_qkv_3way_f32(
    const float* __restrict__ qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int seq, int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * d;
    if (idx >= total) return;
    int s = idx / d;
    int j = idx % d;
    int src_row = s * 3 * d;
    q[idx] = qkv[src_row + j];
    k[idx] = qkv[src_row + d + j];
    v[idx] = qkv[src_row + 2 * d + j];
}

__global__ void split_q_and_gate_f32(
    const float* __restrict__ q_full,
    float* __restrict__ q_only,
    float* __restrict__ q_gate,
    int seq, int n_heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * n_heads * head_dim;
    if (idx >= total) return;
    int d = idx % head_dim;
    int hs = idx / head_dim;             // s * n_heads + h
    int src_base = hs * 2 * head_dim;
    q_only[idx] = q_full[src_base + d];
    q_gate[idx] = q_full[src_base + head_dim + d];
}

// Qwen3.5 / qwen3next gated-delta-net depthwise conv1d FUSED across the seq
// axis: one launch processes all `seq` tokens sequentially, updating the
// rolling conv_state in registers (not global memory) between iterations.
// Replaces 2*seq tiny launches with a single one for prefill. Each thread
// handles one channel `c`. Writes conv_out_full[seq, conv_dim] post-silu and
// updates conv_state to hold the last (conv_kernel-1) input samples.
__global__ void delta_net_conv1d_loop_f32(
    const float* __restrict__ mqkv_full,
    float* __restrict__ conv_state,
    const float* __restrict__ weight,
    float* __restrict__ conv_out_full,
    int seq, int conv_dim, int conv_kernel
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    int K = conv_kernel;
    int km1 = K - 1;

    // Hold the rolling window in registers (assumes K-1 <= 8).
    float ws[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        ws[k] = (k < km1) ? conv_state[k * conv_dim + c] : 0.0f;
    }
    float w[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        w[k] = (k < K) ? weight[c * K + k] : 0.0f;
    }

    for (int t = 0; t < seq; ++t) {
        float new_val = mqkv_full[t * conv_dim + c];
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k < km1) acc += ws[k] * w[k];
        }
        acc += new_val * w[km1];
        conv_out_full[t * conv_dim + c] = acc / (1.0f + expf(-acc));
        // shift window left + append
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k + 1 < km1) ws[k] = ws[k + 1];
        }
        ws[km1 - 1] = new_val;
    }

    // Write back final conv_state.
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < km1) conv_state[k * conv_dim + c] = ws[k];
    }
}

// Qwen3.5 / qwen3next gated-delta-net step FUSED across the seq axis: one
// launch processes all `seq` tokens for one v-head sequentially, with the
// 128x128 state matrix held in registers (one row per thread) for the entire
// loop. Eliminates seq*global-memory state read/write rounds, and gets us
// down to a single launch per layer per direction (conv + step) regardless
// of seq. Output is written into [seq, num_v_heads * head_v_dim].
__global__ void delta_net_step_loop_f32(
    const float* __restrict__ conv_out_full,
    const float* __restrict__ z_full,
    const float* __restrict__ beta_alpha_full,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_norm,
    float* __restrict__ state,
    float* __restrict__ output_full,
    int seq, int num_v_heads, int num_k_heads, int head_v_dim, int head_k_dim,
    int v_per_k, float scale_q, float eps
) {
    int h_v = blockIdx.x;
    int tid = threadIdx.x;
    if (h_v >= num_v_heads) return;
    int h_k = h_v % num_k_heads;
    (void)v_per_k;

    int q_off = h_k * head_k_dim;
    int k_off = num_k_heads * head_k_dim + h_k * head_k_dim;
    int v_off = 2 * num_k_heads * head_k_dim + h_v * head_v_dim;
    int conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;

    __shared__ float q_s[128], k_s[128], v_s[128], z_s[128];
    __shared__ float q_n[128], k_n[128];
    __shared__ float kv_mem[128], delta_v[128], core[128];
    __shared__ float bh, g_t, inv_q, inv_k, core_inv_rms;
    __shared__ float reduce_buf[128];

    // Each of the (head_v_dim) threads holds one row of the per-head state
    // matrix in registers across the seq loop. Saves seq*64KB of state I/O.
    float st_row[128];   // assumes head_v_dim <= 128
    long st_row_off = ((long)h_v * head_v_dim + tid) * head_v_dim;
    if (tid < head_v_dim) {
        #pragma unroll 8
        for (int j = 0; j < 128; ++j) {
            if (j < head_v_dim) st_row[j] = state[st_row_off + j];
        }
    }

    for (int t = 0; t < seq; ++t) {
        long conv_t = (long)t * conv_dim;
        long z_t    = (long)t * num_v_heads * head_v_dim;
        long out_t  = (long)t * num_v_heads * head_v_dim;
        long ba_t   = (long)t * 2 * num_v_heads;

        // Load q, k, v, z into shared mem.
        if (tid < head_k_dim) {
            q_s[tid] = conv_out_full[conv_t + q_off + tid];
            k_s[tid] = conv_out_full[conv_t + k_off + tid];
        }
        if (tid < head_v_dim) {
            v_s[tid] = conv_out_full[conv_t + v_off + tid];
            z_s[tid] = z_full[z_t + h_v * head_v_dim + tid];
        }
        __syncthreads();

        // l2_norm reductions
        float qv = (tid < head_k_dim) ? q_s[tid] * q_s[tid] : 0.0f;
        reduce_buf[tid] = qv; __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
            __syncthreads();
        }
        if (tid == 0) inv_q = rsqrtf(reduce_buf[0] + eps);
        __syncthreads();
        float kv = (tid < head_k_dim) ? k_s[tid] * k_s[tid] : 0.0f;
        reduce_buf[tid] = kv; __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            inv_k = rsqrtf(reduce_buf[0] + eps);
            float bv = beta_alpha_full[ba_t + h_v];
            bh = 1.0f / (1.0f + expf(-bv));
            float ab = beta_alpha_full[ba_t + num_v_heads + h_v] + dt_bias[h_v];
            float ab_sp = (ab > 20.0f) ? ab : log1pf(expf(ab));
            g_t = expf(ab_sp * ssm_a[h_v]);
        }
        __syncthreads();

        if (tid < head_k_dim) {
            q_n[tid] = q_s[tid] * inv_q * scale_q;
            k_n[tid] = k_s[tid] * inv_k;
        }
        __syncthreads();

        // state[i, :] *= g_t and kv_mem[i] = sum_j state[i, j] * k_n[j]
        // (state held in registers per thread)
        if (tid < head_v_dim) {
            float acc = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < 128; ++j) {
                if (j >= head_k_dim) break;
                st_row[j] *= g_t;
                acc += st_row[j] * k_n[j];
            }
            kv_mem[tid] = acc;
        }
        __syncthreads();

        if (tid < head_v_dim) {
            delta_v[tid] = (v_s[tid] - kv_mem[tid]) * bh;
        }
        __syncthreads();

        // state[i, j] += di * k_n[j]; core[i] = sum_j state[i, j] * q_n[j]
        if (tid < head_v_dim) {
            float di = delta_v[tid];
            float acc = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < 128; ++j) {
                if (j >= head_k_dim) break;
                st_row[j] += di * k_n[j];
                acc += st_row[j] * q_n[j];
            }
            core[tid] = acc;
        }
        __syncthreads();

        // RMSNorm of core
        float cv = (tid < head_v_dim) ? core[tid] * core[tid] : 0.0f;
        reduce_buf[tid] = cv; __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
            __syncthreads();
        }
        if (tid == 0) core_inv_rms = rsqrtf(reduce_buf[0] / (float)head_v_dim + eps);
        __syncthreads();

        if (tid < head_v_dim) {
            float normed = core[tid] * core_inv_rms * ssm_norm[tid];
            float zv = z_s[tid];
            float silu_z = zv / (1.0f + expf(-zv));
            output_full[out_t + h_v * head_v_dim + tid] = normed * silu_z;
        }
        __syncthreads();
    }

    // Write final state row back to global memory.
    if (tid < head_v_dim) {
        #pragma unroll 8
        for (int j = 0; j < 128; ++j) {
            if (j < head_v_dim) state[st_row_off + j] = st_row[j];
        }
    }
}

// Qwen3.5 / qwen3next gated-delta-net depthwise conv1d for ONE token. Reads
// the new sample from mqkv_in_full[mqkv_off .. mqkv_off + conv_dim) so the same
// kernel handles both decode (seq=1, mqkv_off=0) and the per-token prefill
// loop (seq>1, mqkv_off=t*conv_dim). conv_state is the rolling window of the
// last (conv_kernel-1) samples per channel; updated in place after consuming
// the new sample. Output is post-silu (one token's worth).
//   mqkv_in_full: [seq, conv_dim] device buffer
//   mqkv_off:     starting float index into mqkv_in_full
//   conv_state:   [conv_kernel-1, conv_dim] rolling window (read-modify-write)
//   weight:       [conv_dim, conv_kernel] depthwise filter
//   conv_out:     [conv_dim] post-silu output (overwritten each call)
__global__ void delta_net_conv1d_decode_f32(
    const float* __restrict__ mqkv_in_full,
    int mqkv_off,
    float* __restrict__ conv_state,
    const float* __restrict__ weight,
    float* __restrict__ conv_out,
    int conv_dim, int conv_kernel
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;

    // Build window: [conv_state[0], conv_state[1], ..., conv_state[K-2], new]
    // and accumulate sum_k window[k] * weight[c, k].
    float acc = 0.0f;
    int k_minus_1 = conv_kernel - 1;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= k_minus_1) break;
        acc += conv_state[k * conv_dim + c] * weight[c * conv_kernel + k];
    }
    float new_val = mqkv_in_full[mqkv_off + c];
    acc += new_val * weight[c * conv_kernel + k_minus_1];

    // silu
    float silu = acc / (1.0f + expf(-acc));
    conv_out[c] = silu;

    // Shift conv_state: [s1, s2, ..., s_{K-2}, new]  (new becomes the last slot;
    // oldest s0 is dropped). Each thread updates only its column.
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= k_minus_1 - 1) break;
        conv_state[k * conv_dim + c] = conv_state[(k + 1) * conv_dim + c];
    }
    conv_state[(k_minus_1 - 1) * conv_dim + c] = new_val;
}

// Qwen3.5 / qwen3next gated-delta-net step for ONE token (decode), running on
// device. One CUDA block per v-head (so launch with grid=(num_v_heads, 1, 1)
// and block=(head_v_dim, 1, 1) — assumes head_v_dim == head_k_dim and ≤ 128
// for the static shared-mem buffers below).
//
//   conv_out:   [conv_dim] f32, already post-silu (output of conv1d above)
//   z_full:     [num_v_heads * head_v_dim] f32, per-token z gate
//   beta_alpha: [2 * num_v_heads] f32, β first then α
//   ssm_a:      [num_v_heads]
//   dt_bias:    [num_v_heads]
//   ssm_norm:   [head_v_dim]
//   state:      [num_v_heads, head_v_dim, head_v_dim] (read-modify-write)
//   output:     [num_v_heads * head_v_dim] f32 (write)
//
// Layout details: conv_out packs [q_all (num_k_heads*head_k_dim) | k_all (...) |
// v_all (num_v_heads*head_v_dim)]; q,k repeat across the v_per_k v-heads that
// share each k-head. z is laid out as [num_k_heads, v_per_k, head_v_dim].
__global__ void delta_net_decode_step_f32(
    const float* __restrict__ conv_out,
    const float* __restrict__ z_full,
    int z_off,
    const float* __restrict__ beta_alpha,
    int ba_off,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_norm,
    float* __restrict__ state,
    float* __restrict__ output,
    int out_off,
    int num_v_heads, int num_k_heads, int head_v_dim, int head_k_dim,
    int v_per_k, float scale_q, float eps
) {
    int h_v = blockIdx.x;
    int tid = threadIdx.x;
    if (h_v >= num_v_heads) return;
    // Unsloth's converter reorders V heads to TILED order:
    //   [K0_V0, K1_V0, ..., K15_V0, K0_V1, ..., K15_V1]
    // so flat index h_v decomposes as h_k = h_v % num_k_heads (NOT h_v / v_per_k).
    // See `_LinearAttentionVReorderBase` in convert_hf_to_gguf.py.
    int h_k = h_v % num_k_heads;
    (void)v_per_k;  // silence unused-arg warning; kept in signature for future use

    int q_off = h_k * head_k_dim;
    int k_off = num_k_heads * head_k_dim + h_k * head_k_dim;
    int v_off = 2 * num_k_heads * head_k_dim + h_v * head_v_dim;

    __shared__ float q_s[128], k_s[128], v_s[128], z_s[128];
    __shared__ float q_n[128], k_n[128];
    __shared__ float kv_mem[128], delta[128], core[128];
    __shared__ float bh, g_t, inv_q, inv_k, core_inv_rms;

    // Load q, k, v, z into shared mem.
    if (tid < head_k_dim) {
        q_s[tid] = conv_out[q_off + tid];
        k_s[tid] = conv_out[k_off + tid];
    }
    if (tid < head_v_dim) {
        v_s[tid] = conv_out[v_off + tid];
        // z is stored in TILED V order (per the unsloth converter), matching the
        // h_v ordering used everywhere else in this kernel — direct indexing
        // with a per-token base offset (z_off=0 for decode, t*4096 for prefill).
        z_s[tid] = z_full[z_off + h_v * head_v_dim + tid];
    }
    __syncthreads();

    // Reduction: sum of squares for q and k across the 128-thread block.
    // Use simple two-pass reduction in shared mem.
    __shared__ float reduce_buf[128];
    // q²
    float qv = (tid < head_k_dim) ? q_s[tid] * q_s[tid] : 0.0f;
    reduce_buf[tid] = qv;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    if (tid == 0) inv_q = rsqrtf(reduce_buf[0] + eps);
    __syncthreads();
    // k²
    float kv = (tid < head_k_dim) ? k_s[tid] * k_s[tid] : 0.0f;
    reduce_buf[tid] = kv;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        inv_k = rsqrtf(reduce_buf[0] + eps);
        // bh = sigmoid(beta[h_v]); β at indices [0, num_v_heads), α at
        // [num_v_heads, 2*num_v_heads) within the per-token slice starting at ba_off.
        float bv = beta_alpha[ba_off + h_v];
        bh = 1.0f / (1.0f + expf(-bv));
        // g_t = exp(softplus(alpha + dt_bias) * ssm_a)
        float ab = beta_alpha[ba_off + num_v_heads + h_v] + dt_bias[h_v];
        float ab_sp = (ab > 20.0f) ? ab : log1pf(expf(ab));
        g_t = expf(ab_sp * ssm_a[h_v]);
    }
    __syncthreads();

    if (tid < head_k_dim) {
        q_n[tid] = q_s[tid] * inv_q * scale_q;
        k_n[tid] = k_s[tid] * inv_k;
    }
    __syncthreads();

    // Each thread handles one row of the [head_v_dim, head_v_dim] state matrix.
    if (tid < head_v_dim) {
        long row_off = ((long)h_v * head_v_dim + tid) * head_v_dim;
        // state[i, :] *= g_t and compute kv_mem[i] = sum_j state[i, j] * k_n[j]
        float acc = 0.0f;
        #pragma unroll 8
        for (int j = 0; j < 128; ++j) {
            if (j >= head_k_dim) break;
            float s = state[row_off + j] * g_t;
            state[row_off + j] = s;
            acc += s * k_n[j];
        }
        kv_mem[tid] = acc;
    }
    __syncthreads();

    if (tid < head_v_dim) {
        delta[tid] = (v_s[tid] - kv_mem[tid]) * bh;
    }
    __syncthreads();

    if (tid < head_v_dim) {
        long row_off = ((long)h_v * head_v_dim + tid) * head_v_dim;
        float di = delta[tid];
        // state[i, j] += di * k_n[j]; then core[i] = sum_j state[i, j] * q_n[j]
        float acc = 0.0f;
        #pragma unroll 8
        for (int j = 0; j < 128; ++j) {
            if (j >= head_k_dim) break;
            float s = state[row_off + j] + di * k_n[j];
            state[row_off + j] = s;
            acc += s * q_n[j];
        }
        core[tid] = acc;
    }
    __syncthreads();

    // RMSNorm of core. Reuse reduce_buf.
    float cv = (tid < head_v_dim) ? core[tid] * core[tid] : 0.0f;
    reduce_buf[tid] = cv;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    if (tid == 0) core_inv_rms = rsqrtf(reduce_buf[0] / (float)head_v_dim + eps);
    __syncthreads();

    if (tid < head_v_dim) {
        float normed = core[tid] * core_inv_rms * ssm_norm[tid];
        float zv = z_s[tid];
        float silu_z = zv / (1.0f + expf(-zv));
        output[out_off + h_v * head_v_dim + tid] = normed * silu_z;
    }
}

}  // extern "C"

// VENDORED-LOCAL: GPU-02 — test-only busy-wait kernel. Occupies exactly one
// block for `cycles` clock ticks so overlap tests can keep the compute
// stream busy with negligible SM/HBM footprint (a saturating GEMM throttles
// copy-engine H2D on WDDM and confuses wall-clock overlap assertions).
extern "C" __global__ void spin_f32(float* __restrict__ out, long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) { }
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = 1.0f;
}

// ===================== VENDORED-LOCAL: MOE-01 / MOE-02 =====================
// GPU-resident MoE routing and grouped expert execution (docs/ROADMAP.md
// Phase 3). The launch contract per MoE layer at decode (seq=1) is:
//   router gemv → moe_topk_softmax_f32 → moe_gate_up_act_<q> →
//   moe_down_scale_<q> → moe_reduce_slots_f32
// with routing ids/weights never leaving the device (resident path) or a
// compact ids-only mailbox D2H (streaming path's miss list).
//
// Numerics contract: the row-dot helpers below replicate the existing coop
// GEMV kernels' partitioning EXACTLY where those kernels apply (Q4_K with
// total_sb % 32 == 0, Q6_K always, Q8_0 always), so a grouped expert's
// intermediate is bit-identical to the reference linear_q → silu_mul_split
// chain. For Q4_K rows whose sub-block count does not divide 32 (e.g. the
// Qwen3-30B down projection with K=768) a deterministic strided partition is
// used instead — same per-sub-block math, different cross-thread association
// only (validated token-id-exact on the real models).

extern "C" {

// MOE-01: top-k + softmax over one row of router logits per block.
// Grid (seq), block (256). Dynamic shared: n_experts floats (working copy).
// Selection is k block-wide argmax passes; ties break toward the lower index
// (matches the CPU sampler's earliest-max convention; the CPU partial-sort
// reference is unspecified on exact f32 ties, which random/trained logits
// effectively never produce). Weights: softmax over the selected logits,
// computed serially by thread 0 in slot order — same association as the host
// reference. Guards (host side): 1 <= top_k <= 128, n_experts <= 8192.
__global__ void moe_topk_softmax_f32(const float* __restrict__ logits,
                                      unsigned int* __restrict__ out_ids,
                                      float* __restrict__ out_w,
                                      int n_experts, int top_k) {
    int row = blockIdx.x;
    const float* x = logits + (long)row * n_experts;
    extern __shared__ float s_val[];      // [n_experts] working copy of the row
    __shared__ float red_v[256];
    __shared__ int   red_i[256];
    __shared__ float sel[128];            // selected logits, slot order
    int tid = threadIdx.x;

    for (int i = tid; i < n_experts; i += 256) s_val[i] = x[i];
    __syncthreads();

    for (int s = 0; s < top_k; ++s) {
        float bv = -INFINITY;
        int   bi = 0x7fffffff;
        for (int i = tid; i < n_experts; i += 256) {
            float v = s_val[i];
            if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
        red_v[tid] = bv;
        red_i[tid] = bi;
        __syncthreads();
        #pragma unroll
        for (int off = 128; off > 0; off >>= 1) {
            if (tid < off) {
                float ov = red_v[tid + off];
                int   oi = red_i[tid + off];
                if (ov > red_v[tid] || (ov == red_v[tid] && oi < red_i[tid])) {
                    red_v[tid] = ov;
                    red_i[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            sel[s] = red_v[0];
            out_ids[row * top_k + s] = (unsigned int)red_i[0];
            s_val[red_i[0]] = -INFINITY;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = sel[0];
        float sum = 0.0f;
        for (int s = 0; s < top_k; ++s) sum += expf(sel[s] - mx);
        for (int s = 0; s < top_k; ++s) out_w[row * top_k + s] = expf(sel[s] - mx) / sum;
    }
}

// ---- per-quant row-dot helpers (warp-cooperative, acc by reference) --------

// Q4_K: identical loop nest + header caching as linear_q4_k_gemv_coop_f32
// when total_sb % 32 == 0 (bit-exact with it); deterministic strided
// sub-block partition otherwise.
__device__ __forceinline__ void moe_q4k_row_accum(float* acc,
        const float* __restrict__ x, const unsigned char* __restrict__ w_row,
        int K, int t) {
    int n_blocks = K / 256;
    int total_sb = n_blocks * 8;
    if (total_sb % 32 == 0) {
        int sb_per_thd = total_sb / 32;
        int sb_start   = t * sb_per_thd;
        int sb_end     = sb_start + sb_per_thd;
        int last_b = -1;
        float d_sb = 0.0f, min_sb = 0.0f;
        const unsigned char* qs_sb = nullptr;
        const unsigned char* sc_sb = nullptr;
        for (int sb = sb_start; sb < sb_end; ++sb) {
            int b  = sb >> 3;
            int is = sb & 7;
            if (b != last_b) {
                const unsigned char* bp = w_row + b * 144;
                unsigned short d_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
                unsigned short m_bits = (unsigned short)bp[2] | ((unsigned short)bp[3] << 8);
                d_sb   = f16_to_f32(d_bits);
                min_sb = f16_to_f32(m_bits);
                sc_sb  = bp + 4;
                qs_sb  = bp + 16;
                last_b = b;
            }
            unsigned char sc_byte, mm_byte;
            q4k_unpack_scale_min(is, sc_sb, &sc_byte, &mm_byte);
            float dq = d_sb   * (float)sc_byte;
            float mq = min_sb * (float)mm_byte;
            int qs_off = (is >> 1) * 32;
            const unsigned char* qsp = qs_sb + qs_off;
            const float* xp = x + b * 256 + is * 32;
            if ((is & 1) == 0) {
                #pragma unroll
                for (int l = 0; l < 32; ++l) {
                    float qv = (float)(qsp[l] & 0x0F);
                    *acc += xp[l] * (dq * qv - mq);
                }
            } else {
                #pragma unroll
                for (int l = 0; l < 32; ++l) {
                    float qv = (float)((qsp[l] >> 4) & 0x0F);
                    *acc += xp[l] * (dq * qv - mq);
                }
            }
        }
    } else {
        // Strided fallback: thread t takes sub-blocks t, t+32, ... — no
        // header caching (consecutive sub-blocks rarely share a super-block),
        // same per-sub-block math as above.
        for (int sb = t; sb < total_sb; sb += 32) {
            int b  = sb >> 3;
            int is = sb & 7;
            const unsigned char* bp = w_row + b * 144;
            unsigned short d_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
            unsigned short m_bits = (unsigned short)bp[2] | ((unsigned short)bp[3] << 8);
            float d_sb   = f16_to_f32(d_bits);
            float min_sb = f16_to_f32(m_bits);
            unsigned char sc_byte, mm_byte;
            q4k_unpack_scale_min(is, bp + 4, &sc_byte, &mm_byte);
            float dq = d_sb   * (float)sc_byte;
            float mq = min_sb * (float)mm_byte;
            const unsigned char* qsp = bp + 16 + (is >> 1) * 32;
            const float* xp = x + b * 256 + is * 32;
            if ((is & 1) == 0) {
                #pragma unroll
                for (int l = 0; l < 32; ++l) {
                    float qv = (float)(qsp[l] & 0x0F);
                    *acc += xp[l] * (dq * qv - mq);
                }
            } else {
                #pragma unroll
                for (int l = 0; l < 32; ++l) {
                    float qv = (float)((qsp[l] >> 4) & 0x0F);
                    *acc += xp[l] * (dq * qv - mq);
                }
            }
        }
    }
}

// Q6_K: identical partitioning to linear_q6_k_gemv_coop_f32 (thread t owns
// inner step l=t in both outer iters of every super-block) — bit-exact.
__device__ __forceinline__ void moe_q6k_row_accum(float* acc,
        const float* __restrict__ x, const unsigned char* __restrict__ w_row,
        int K, int t) {
    int n_blocks = K / 256;
    int l = t;
    int is = l >> 4;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 210;
        const unsigned char* ql = bp + 0;
        const unsigned char* qh = bp + 128;
        const signed char*   sc = (const signed char*)(bp + 192);
        unsigned short d_bits = (unsigned short)bp[208] | ((unsigned short)bp[209] << 8);
        float d = f16_to_f32(d_bits);
        const float* xb = x + b * 256;
        {
            int q1 = (int)((ql[l]      & 0x0F) | (((qh[l] >> 0) & 0x3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x3) << 4)) - 32;
            int q3 = (int)((ql[l]      >> 4)   | (((qh[l] >> 4) & 0x3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 0x3) << 4)) - 32;
            *acc += xb[l]      * d * (float)sc[is + 0] * (float)q1;
            *acc += xb[l + 32] * d * (float)sc[is + 2] * (float)q2;
            *acc += xb[l + 64] * d * (float)sc[is + 4] * (float)q3;
            *acc += xb[l + 96] * d * (float)sc[is + 6] * (float)q4;
        }
        {
            int q1 = (int)((ql[64 + l]      & 0x0F) | (((qh[32 + l] >> 0) & 0x3) << 4)) - 32;
            int q2 = (int)((ql[64 + l + 32] & 0x0F) | (((qh[32 + l] >> 2) & 0x3) << 4)) - 32;
            int q3 = (int)((ql[64 + l]      >> 4)   | (((qh[32 + l] >> 4) & 0x3) << 4)) - 32;
            int q4 = (int)((ql[64 + l + 32] >> 4)   | (((qh[32 + l] >> 6) & 0x3) << 4)) - 32;
            *acc += xb[128 + l]      * d * (float)sc[8 + is + 0] * (float)q1;
            *acc += xb[128 + l + 32] * d * (float)sc[8 + is + 2] * (float)q2;
            *acc += xb[128 + l + 64] * d * (float)sc[8 + is + 4] * (float)q3;
            *acc += xb[128 + l + 96] * d * (float)sc[8 + is + 6] * (float)q4;
        }
    }
}

// Q8_0: identical partitioning to linear_q8_0_gemv_coop_f32 (thread t owns
// element t of every 32-wide block) — bit-exact.
__device__ __forceinline__ void moe_q8_0_row_accum(float* acc,
        const float* __restrict__ x, const unsigned char* __restrict__ w_row,
        int K, int t) {
    int n_blocks = K / 32;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* bp = w_row + b * 34;
        unsigned short scale_bits = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
        float d = f16_to_f32(scale_bits);
        const signed char* qs = (const signed char*)(bp + 2);
        const float* xb = x + b * 32;
        *acc += xb[t] * (float)qs[t] * d;
    }
}

// Standard butterfly warp reduction of one register value.
__device__ __forceinline__ float moe_warp_reduce(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

// MOE-02a: fused gate_up matvec + activation for ONE token's k experts.
// Grid (ceil(ff/8), k), block (32, 8): each warp computes one intermediate
// element act[slot, j] = act_fn(gate_j · x) * (up_j · x) for the expert
// routed in slot `blockIdx.y`. `wtab` holds one device pointer per table
// entry and is indexed as `wtab[idx_base + expert_ids[slot]]`: a resident
// layer uploads one static [gate_up n_experts | down n_experts] table and
// passes idx_base 0 / n_experts; a streaming layer-forward uploads the same
// combined layout for its k staged experts (unrouted entries are never
// read). use_gelu: 0 = silu (SwiGLU), 1 = tanh-approx gelu (GeGLU, Gemma 4
// MoE). Activation formulas replicate silu_mul_split_f32 /
// gelu_approx_mul_split_f32 exactly.
#define MOE_GATE_UP_ACT(NAME, ACCUM, ROW_BYTES_EXPR)                                \
__global__ void NAME(const float* __restrict__ x,                                   \
                     const unsigned long long* __restrict__ wtab,                   \
                     const unsigned int* __restrict__ expert_ids,                   \
                     float* __restrict__ act,                                       \
                     int ff, int K, int use_gelu, int idx_base) {                   \
    int slot = blockIdx.y;                                                          \
    int j = blockIdx.x * blockDim.y + threadIdx.y;                                  \
    if (j >= ff) return;                                                            \
    int t = threadIdx.x;                                                            \
    const unsigned char* w =                                                        \
        (const unsigned char*)wtab[idx_base + expert_ids[slot]];                    \
    int row_bytes = (ROW_BYTES_EXPR);                                               \
    float accg = 0.0f, accu = 0.0f;                                                 \
    ACCUM(&accg, x, w + (long)j * row_bytes, K, t);                                 \
    ACCUM(&accu, x, w + (long)(ff + j) * row_bytes, K, t);                          \
    accg = moe_warp_reduce(accg);                                                   \
    accu = moe_warp_reduce(accu);                                                   \
    if (t == 0) {                                                                   \
        float a;                                                                    \
        if (use_gelu) {                                                             \
            const float SQRT_2_OVER_PI = 0.7978845608028654f;                       \
            const float COEFF = 0.044715f;                                          \
            float inner = SQRT_2_OVER_PI * (accg + COEFF * accg * accg * accg);     \
            a = 0.5f * accg * (1.0f + tanhf(inner));                                \
        } else {                                                                    \
            a = accg / (1.0f + __expf(-accg));                                      \
        }                                                                           \
        act[slot * ff + j] = a * accu;                                              \
    }                                                                               \
}

MOE_GATE_UP_ACT(moe_gate_up_act_q4_k_f32, moe_q4k_row_accum,  (K / 256) * 144)
MOE_GATE_UP_ACT(moe_gate_up_act_q6_k_f32, moe_q6k_row_accum,  (K / 256) * 210)
MOE_GATE_UP_ACT(moe_gate_up_act_q8_0_f32, moe_q8_0_row_accum, (K / 32) * 34)

// MOE-02b: fused down matvec + route-scale into per-slot partials.
// Grid (ceil(hidden/8), k), block (32, 8): each warp computes
// partial[slot, i] = (down_i · act[slot]) * weights[slot] [* scales[expert]].
// `wtab` follows the same idx_base convention as gate_up. `scales` is read
// only when has_scales != 0 (Gemma 4 MoE per-expert down scale); pass any
// valid f32 pointer otherwise.
#define MOE_DOWN_SCALE(NAME, ACCUM, ROW_BYTES_EXPR)                                 \
__global__ void NAME(const float* __restrict__ act,                                 \
                     const unsigned long long* __restrict__ wtab,                   \
                     const unsigned int* __restrict__ expert_ids,                   \
                     const float* __restrict__ weights,                             \
                     const float* __restrict__ scales,                              \
                     float* __restrict__ partial,                                   \
                     int hidden, int K, int has_scales, int idx_base) {             \
    int slot = blockIdx.y;                                                          \
    int i = blockIdx.x * blockDim.y + threadIdx.y;                                  \
    if (i >= hidden) return;                                                        \
    int t = threadIdx.x;                                                            \
    int expert = (int)expert_ids[slot];                                             \
    const unsigned char* w =                                                        \
        (const unsigned char*)wtab[idx_base + expert];                              \
    int row_bytes = (ROW_BYTES_EXPR);                                               \
    float acc = 0.0f;                                                               \
    ACCUM(&acc, act + slot * K, w + (long)i * row_bytes, K, t);                     \
    acc = moe_warp_reduce(acc);                                                     \
    if (t == 0) {                                                                   \
        float sc = weights[slot];                                                   \
        if (has_scales) sc *= scales[expert];                                       \
        partial[slot * hidden + i] = acc * sc;                                      \
    }                                                                               \
}

MOE_DOWN_SCALE(moe_down_scale_q4_k_f32, moe_q4k_row_accum,  (K / 256) * 144)
MOE_DOWN_SCALE(moe_down_scale_q6_k_f32, moe_q6k_row_accum,  (K / 256) * 210)
MOE_DOWN_SCALE(moe_down_scale_q8_0_f32, moe_q8_0_row_accum, (K / 32) * 34)

// MOE-02c: fixed-order reduction of the per-slot partials into the output.
// out[i] = ((0 + partial[0,i]) + partial[1,i]) + ... — the same association
// as the reference loop's sequential add_to_axis0_range_scaled calls, so a
// bit-exact partial yields a bit-exact MoE output.
__global__ void moe_reduce_slots_f32(const float* __restrict__ partial,
                                      float* __restrict__ out, int k, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    float acc = 0.0f;
    for (int s = 0; s < k; ++s) acc += partial[s * hidden + i];
    out[i] = acc;
}

}  // extern "C" (MOE-01/MOE-02)
"#;

pub const KERNEL_NAMES: &[&str] = &[
    "linear_q8_0_f32",
    "linear_q8_0_gemv_coop_f32",
    "linear_iq4_nl_f32",
    "linear_iq4_xs_f32",
    "linear_q2_k_f32",
    "linear_q3_k_f32",
    "linear_q4_0_f32",
    "linear_q4_1_f32",
    "linear_q4_k_f32",
    "linear_q5_k_f32",
    "linear_q5_k_gemv_coop_f32",
    "linear_q5_0_f32",
    "linear_q5_0_gemv_coop_f32",
    "linear_q5_1_f32",
    "linear_q5_1_gemv_coop_f32",
    "linear_q6_k_f32",
    "linear_q6_k_gemv_coop_f32",
    "linear_f32",
    "linear_f32_gemv_coop",
    "rmsnorm_f32",
    "add_inplace_then_rmsnorm_f32",
    "rmsnorm_no_scale_f32",
    "layer_norm_f32",
    "add_bias_last_f32",
    "softmax_last_f32",
    "silu_f32",
    "silu_mul_f32",
    "sigmoid_f32",
    "mul_sigmoid_inplace_f32",
    "gelu_approx_f32",
    "gelu_approx_mul_f32",
    "silu_mul_split_f32",
    // VENDORED-LOCAL: GLM-5.3-Flash.
    "swiglu_clamped_f32",
    "swiglu_clamped_split_f32",
    // VENDORED-LOCAL: GLM-5.3-Flash.
    "batched_gemv_f32",
    // VENDORED-LOCAL: GLM-5.3-Flash.
    "kda_delta_step_f32",
    "gelu_approx_mul_split_f32",
    "tanh_inplace_f32",
    "gaussian_topk_inplace_f32",
    "altup_predict_f32",
    "altup_correct_f32",
    "add_inplace_f32",
    "mul_inplace_f32",
    "mul_scalar_inplace_f32",
    "mul_inplace_broadcast_last_f32",
    "mul_inplace_broadcast_axis0_f32",
    "add_to_axis0_range_f32",
    "add_to_axis0_range_scaled_f32",
    "slice_axis1_2d_f32",
    "rope_f32",
    "rope_partial_neox_f32",
    "repeat_kv_f32",
    "argmax_last_f32",
    "embed_lookup_f32",
    "bmm_qkt_f32",
    "bmm_av_f32",
    "attention_f32",
    "delta_net_conv1d_decode_f32",
    "delta_net_decode_step_f32",
    "delta_net_conv1d_loop_f32",
    "delta_net_step_loop_f32",
    "split_q_and_gate_f32",
    "split_qkv_3way_f32",
    "linear_q4_k_gemv_coop_f32",
    // VENDORED-LOCAL: GPU-02 — test-only spin kernel (see KERNEL_SRC).
    "spin_f32",
    // VENDORED-LOCAL: MOE-01/MOE-02 — GPU routing + grouped MoE kernels.
    "moe_topk_softmax_f32",
    "moe_gate_up_act_q4_k_f32",
    "moe_gate_up_act_q6_k_f32",
    "moe_gate_up_act_q8_0_f32",
    "moe_down_scale_q4_k_f32",
    "moe_down_scale_q6_k_f32",
    "moe_down_scale_q8_0_f32",
    "moe_reduce_slots_f32",
];
