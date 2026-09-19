// DeepSeek-V4.1 kernels, compiled at runtime by NVRTC (see gpu.rs).
//
// Every kernel mirrors a CPU function in the `dsv41` crate, which is itself
// validated against the reference model; the CPU function is the test oracle
// for the kernel (tests/kernels.rs). Conventions that make that comparison
// tight:
// - activations are f32 buffers holding the values the reference holds, so
//   "bf16" means "rounded through bf16 at the same point" (to_bf16 below);
// - fp8 / fp4 / e8m0 conversions are hand-rolled bit manipulation (no
//   cuda_fp8.h), and the quantizers are bit-exact with the CPU versions;
// - compiled with --fmad=false: the CPU path never contracts a*b+c, so the
//   GPU must not either;
// - reductions differ in order from the CPU's sequential sums, so GEMV /
//   attention / norms match to tolerance, not bits.

#define INF __int_as_float(0x7f800000)

extern "C" {

// ---------------------------------------------------------------- scalars

__device__ __forceinline__ float to_bf16(float x) {
    unsigned int b = __float_as_uint(x);
    if ((b & 0x7fffffffu) > 0x7f800000u) return __uint_as_float((b | 0x00400000u) & 0xffff0000u);
    b += 0x7fffu + ((b >> 16) & 1u);
    return __uint_as_float(b & 0xffff0000u);
}

__device__ __forceinline__ float bf16_val(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }

__device__ __forceinline__ float e8m0_val(unsigned char b) {
    if (b == 0) return __uint_as_float(0x00400000u);
    if (b == 255) return __uint_as_float(0x7fc00000u);
    return __uint_as_float(((unsigned int)b) << 23);
}

__device__ __forceinline__ float fp8_val(unsigned char b) {
    unsigned int e = (b >> 3) & 0xfu, m = b & 0x7u;
    float v;
    if (e == 0) v = (float)m * 0.001953125f;  // m/8 * 2^-6
    else if (e == 15 && m == 7) v = __uint_as_float(0x7fc00000u);
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80) ? -v : v;
}

// 2^ceil(log2(x)) from the exponent bits, as the reference fast_round_scale
__device__ __forceinline__ float pow2_ceil(float x) {
    unsigned int bits = __float_as_uint(x);
    int e = (int)((bits >> 23) & 0xffu) - 127 + ((bits & 0x7fffffu) != 0u);
    return __uint_as_float((unsigned int)(e + 127) << 23);
}

// nearest e4m3 value (ties to even); v already clamped to +-448
__device__ __forceinline__ float round_e4m3(float v) {
#if __CUDA_ARCH__ >= 890
    // the hardware conversion is the same round-to-nearest-even onto the e4m3
    // grid (subnormals included, saturating at 448), then back exactly
    unsigned short q8;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(q8) : "f"(0.0f), "f"(v));
    float r;
    asm("{ .reg .b32 h2; .reg .f16 lo, hi; cvt.rn.f16x2.e4m3x2 h2, %1; mov.b32 {lo, hi}, h2; cvt.f32.f16 %0, lo; }"
        : "=f"(r) : "h"(q8));
    return r;
#else
    float a = fabsf(v), q;
    if (a < 0.015625f) {  // below 2^-6: subnormal grid, step 2^-9
        q = rintf(a * 512.0f) * 0.001953125f;
    } else {
        int e;
        frexpf(a, &e);  // a = m * 2^e, m in [0.5, 1): step is 2^(e-4)
        q = ldexpf(rintf(ldexpf(a, 4 - e)), e - 4);
    }
    return copysignf(fminf(q, 448.0f), v);
#endif
}

// nearest e2m1 value (ties to even); v already clamped to +-6
__device__ __forceinline__ float round_e2m1(float v) {
    float a = fabsf(v), q;
    if (a < 2.0f) q = rintf(a * 2.0f) * 0.5f;
    else if (a < 4.0f) q = rintf(a);
    else q = fminf(rintf(a * 0.5f) * 2.0f, 6.0f);
    return copysignf(q, v);
}

// fp8_val without branches (same values; decodes many bytes per lane)
__device__ __forceinline__ float fp8_fast(unsigned int b) {
    unsigned int e = (b >> 3) & 0xfu, m = b & 0x7u;
    float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
    v = (e == 15u && m == 7u) ? __uint_as_float(0x7fc00000u) : v;
    return __uint_as_float(__float_as_uint(v) | ((b & 0x80u) << 24));
}
// Two e4m3 bytes (low byte first) to f32, exactly: one hardware conversion on
// sm_89+ (e4m3 -> f16 is exact, f16 -> f32 too), fp8_fast elsewhere.
__device__ __forceinline__ void fp8x2(unsigned int two, float& a, float& b) {
#if __CUDA_ARCH__ >= 890
    unsigned int h2;
    unsigned short in = (unsigned short)(two & 0xffffu);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(in));
    asm("{\n\t.reg .f16 lo, hi;\n\tmov.b32 {lo, hi}, %2;\n\tcvt.f32.f16 %0, lo;\n\tcvt.f32.f16 %1, hi;\n\t}"
        : "=f"(a), "=f"(b) : "r"(h2));
#else
    a = fp8_fast(two & 0xffu);
    b = fp8_fast((two >> 8) & 0xffu);
#endif
}
__device__ __forceinline__ bool aligned16(const void* p) { return (((unsigned long long)p) & 15ull) == 0ull; }
// 32 consecutive floats, as 8 float4 loads when aligned
__device__ __forceinline__ void load32(const float* p, bool vec, float* o) {
    if (vec) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            float4 v = *(const float4*)(p + 4 * i);
            o[4 * i] = v.x; o[4 * i + 1] = v.y; o[4 * i + 2] = v.z; o[4 * i + 3] = v.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 32; i++) o[i] = p[i];
    }
}
// byte i of a 16-byte vector
__device__ __forceinline__ unsigned int byte_of(const uint4& q, int i) {
    unsigned int w = i < 4 ? q.x : i < 8 ? q.y : i < 12 ? q.z : q.w;
    return (w >> (8 * (i & 3))) & 0xffu;
}
__device__ __constant__ float FP4[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// ---------------------------------------------------------------- quantizers

// formats::fake_quant_fp8 in place: per 32 values, scale pow2_ceil(amax/448)
// (amax >= 1e-4), e4m3 round trip. One warp per block; n % 32 == 0.
__global__ void act_quant_fp8(float* x, int n) {
    int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (blk * 32 >= n) return;
    float v = x[blk * 32 + lane];
    float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
    float s = pow2_ceil(amax * (1.0f / 448.0f));
    x[blk * 32 + lane] = round_e4m3(fminf(fmaxf(v / s, -448.0f), 448.0f)) * s;
}

// act_quant_fp8 out of place: y = quantized x (saves the copy an in-place
// quantize of a still-needed x would take).
__global__ void act_quant_fp8_to(const float* x, float* y, int n) {
    int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (blk * 32 >= n) return;
    float v = x[blk * 32 + lane];
    float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
    float s = pow2_ceil(amax * (1.0f / 448.0f));
    y[blk * 32 + lane] = round_e4m3(fminf(fmaxf(v / s, -448.0f), 448.0f)) * s;
}
// attention::fake_quant_fp4_inplace: `block` in {16, 32}; kind 0 = e8m0
// scale (indexer), 1 = e4m3 scale (compressed KV). Output rounded to bf16.
// One warp per block; lanes >= block idle.
__global__ void act_quant_fp4(float* x, int n, int block, int kind) {
    int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (blk * block >= n) return;
    bool on = lane < block;
    float v = on ? x[blk * block + lane] : 0.0f;
    float amax = warp_max(fabsf(v));
    float s = kind == 0 ? pow2_ceil(fmaxf(amax, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f))
                        : round_e4m3(fmaxf(amax, 6.0f * 0.001953125f) / 6.0f);
    if (on) x[blk * block + lane] = to_bf16(round_e2m1(fminf(fmaxf(v / s, -6.0f), 6.0f)) * s);
}

// ---------------------------------------------------------------- GEMV

// y[t][row] = sum_k x[t][k] * W[row][k], one warp per output row, `nt`
// tokens handled 8 at a time with per-lane accumulators. `group_rows > 0`
// makes W block-diagonal: rows [g*group_rows, (g+1)*group_rows) read input
// columns [g*k, (g+1)*k) of an x row that is `x_stride` wide (wo_a).
// round: 1 = round outputs to bf16.
#define GEMV_TOK 8

// fp8 e4m3 weights with one e8m0 scale per 32x32 tile; x already fp8-quantized.
__global__ void gemv_fp8(const float* x, const unsigned char* w, const unsigned char* s, float* y,
                         int n, int k, int nt, int round) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    int kb = (k + 31) / 32;
    const unsigned char* wr = w + (long)row * k;
    const unsigned char* sr = s + (long)(row / 32) * kb;
    // a lane owns whole 32-blocks (b = lane, lane + 32, ...): loaded as two
    // 16-byte vectors, summed in the same order as the byte loop
    bool wv16 = (k % 32 == 0) && aligned16(wr), xv = (k % 4 == 0) && aligned16(x);
    for (int t0 = 0; t0 < nt; t0 += GEMV_TOK) {
        float acc[GEMV_TOK] = {0};
        int tn = min(GEMV_TOK, nt - t0);
        for (int b = lane; b < kb; b += 32) {
            float sc = e8m0_val(sr[b]);
            int c0 = b * 32, c1 = min(c0 + 32, k);
            float part[GEMV_TOK] = {0};
            if (wv16) {
                uint4 q0 = *(const uint4*)(wr + c0), q1 = *(const uint4*)(wr + c0 + 16);
                float wf[32];
                unsigned int words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    fp8x2(words[i], wf[4 * i], wf[4 * i + 1]);
                    fp8x2(words[i] >> 16, wf[4 * i + 2], wf[4 * i + 3]);
                }
                for (int t = 0; t < tn; t++) {
                    float xf[32];
                    load32(x + (long)(t0 + t) * k + c0, xv, xf);
#pragma unroll
                    for (int c = 0; c < 32; c++) part[t] += xf[c] * wf[c];
                }
            } else {
                for (int c = c0; c < c1; c++) {
                    float wv = fp8_val(wr[c]);
                    for (int t = 0; t < tn; t++) part[t] += x[(long)(t0 + t) * k + c] * wv;
                }
            }
            for (int t = 0; t < tn; t++) acc[t] += part[t] * sc;
        }
        for (int t = 0; t < tn; t++) {
            float v = warp_sum(acc[t]);
            if (lane == 0) y[(long)(t0 + t) * n + row] = round ? to_bf16(v) : v;
        }
    }
}

// The shared expert's gate and up for one token, SwiGLU in the epilogue:
// h[r] = bf16(silu(min(g, lim)) * clamp(u, lim)) with g, u = bf16 rows r of
// w1 / w3 (fp8, 32x32 tile scales) against the fp8-quantized xq. Each dot
// is gemv_fp8's (a lane owns whole 32-blocks, same order), so this is
// gemv_fp8 twice then swiglu, bit for bit, in one launch. One warp per row;
// needs k % 32 == 0 and 16-byte aligned rows.
__global__ void shared_gate_up(const float* xq, const unsigned char* w1, const unsigned char* s1, const unsigned char* w3,
                               const unsigned char* s3, float* h, int n, int k, float lim) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    int kb = k / 32;
    const unsigned char* ra = w1 + (long)row * k;
    const unsigned char* rb = w3 + (long)row * k;
    const unsigned char* sa = s1 + (long)(row / 32) * kb;
    const unsigned char* sb = s3 + (long)(row / 32) * kb;
    bool xv = aligned16(xq);
    float ag = 0.0f, au = 0.0f;
    for (int b = lane; b < kb; b += 32) {
        float xf[32];
        load32(xq + b * 32, xv, xf);
        uint4 g0 = *(const uint4*)(ra + b * 32), g1 = *(const uint4*)(ra + b * 32 + 16);
        uint4 u0 = *(const uint4*)(rb + b * 32), u1 = *(const uint4*)(rb + b * 32 + 16);
        float wg[32], wu[32];
        unsigned int gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
        unsigned int uw[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
        for (int i = 0; i < 8; i++) {
            fp8x2(gw[i], wg[4 * i], wg[4 * i + 1]);
            fp8x2(gw[i] >> 16, wg[4 * i + 2], wg[4 * i + 3]);
            fp8x2(uw[i], wu[4 * i], wu[4 * i + 1]);
            fp8x2(uw[i] >> 16, wu[4 * i + 2], wu[4 * i + 3]);
        }
        float pg = 0.0f, pu = 0.0f;
#pragma unroll
        for (int c = 0; c < 32; c++) {
            pg += xf[c] * wg[c];
            pu += xf[c] * wu[c];
        }
        ag += pg * e8m0_val(sa[b]);
        au += pu * e8m0_val(sb[b]);
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) {
        float g = to_bf16(ag), u = to_bf16(au);
        if (lim > 0.0f) {
            u = fminf(fmaxf(u, -lim), lim);
            g = fminf(g, lim);
        }
        h[row] = to_bf16(g / (1.0f + expf(-g)) * u);
    }
}

// One-token gemv_fp8 with the activation quantizer fused in. Each block puts
// x into shared memory, fp8-quantized with act_quant_fp8's exact arithmetic
// when `quant` (so the result is bit-identical to act_quant_fp8 then
// gemv_fp8), then its warps take rows two at a time (r and r + nw: one
// shared-memory read of x serves both, and twice the loads are in flight),
// a lane owning whole 32-blocks (b = lane, lane + 32, ...) summed in
// gemv_fp8's order. Needs k % 32 == 0 and 16-byte aligned rows; k floats of
// shared memory.
__device__ __forceinline__ void fp8_block(const unsigned char* wr, int b, float* wf) {
    uint4 q0 = *(const uint4*)(wr + b * 32), q1 = *(const uint4*)(wr + b * 32 + 16);
    unsigned int words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
    for (int i = 0; i < 8; i++) {
        fp8x2(words[i], wf[4 * i], wf[4 * i + 1]);
        fp8x2(words[i] >> 16, wf[4 * i + 2], wf[4 * i + 3]);
    }
}

__global__ void gemv_fp8_token(const float* x, const unsigned char* w, const unsigned char* s, float* y,
                            int n, int k, int rows_per_block, int round, int quant) {
    extern __shared__ float xs[];
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31, nw = blockDim.x / 32;
    int kb = k / 32;
#pragma unroll 4
    for (int b = warp; b < kb; b += nw) {
        float v = x[b * 32 + lane];
        if (quant) {
            float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
            float sc = pow2_ceil(amax * (1.0f / 448.0f));
            v = round_e4m3(fminf(fmaxf(v / sc, -448.0f), 448.0f)) * sc;
        }
        xs[b * 32 + lane] = v;
    }
    __syncthreads();
    int r0 = blockIdx.x * rows_per_block, r1 = min(r0 + rows_per_block, n);
    for (int ra = r0 + warp; ra < r1; ra += 2 * nw) {
        int rb = ra + nw;
        bool two = rb < r1;
        const unsigned char* wa = w + (long)ra * k;
        const unsigned char* wb = w + (long)(two ? rb : ra) * k;
        const unsigned char* sa = s + (long)(ra / 32) * kb;
        const unsigned char* sb = s + (long)((two ? rb : ra) / 32) * kb;
        float acca = 0.0f, accb = 0.0f;
#pragma unroll 2
        for (int b = lane; b < kb; b += 32) {
            const float4* xv = (const float4*)(xs + b * 32);
            float xf[32], fa[32], fb[32];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 v = xv[i];
                xf[4 * i] = v.x; xf[4 * i + 1] = v.y; xf[4 * i + 2] = v.z; xf[4 * i + 3] = v.w;
            }
            fp8_block(wa, b, fa);
            fp8_block(wb, b, fb);
            float pa = 0.0f, pb = 0.0f;
#pragma unroll
            for (int c = 0; c < 32; c++) {
                pa += xf[c] * fa[c];
                pb += xf[c] * fb[c];
            }
            acca += pa * e8m0_val(sa[b]);
            accb += pb * e8m0_val(sb[b]);
        }
        float va = warp_sum(acca), vb = warp_sum(accb);
        if (lane == 0) {
            y[ra] = round ? to_bf16(va) : va;
            if (two) y[rb] = round ? to_bf16(vb) : vb;
        }
    }
}

__global__ void gemv_bf16(const float* x, const unsigned short* w, float* y, int n, int k, int nt, int round,
                          int group_rows, int x_stride) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    const unsigned short* wr = w + (long)row * k;
    int xoff = group_rows > 0 ? (row / group_rows) * k : 0;
    int xs = group_rows > 0 ? x_stride : k;
    // 8 weights per lane per load (one 16-byte vector) when rows allow it
    bool vec = (k % 8 == 0) && aligned16(wr) && aligned16(x) && (xs % 4 == 0) && (xoff % 4 == 0);
    for (int t0 = 0; t0 < nt; t0 += GEMV_TOK) {
        float acc[GEMV_TOK] = {0};
        int tn = min(GEMV_TOK, nt - t0);
        if (vec) {
            for (int c = lane * 8; c < k; c += 256) {
                uint4 q = *(const uint4*)(wr + c);
                float wf[8] = {bf16_val(q.x & 0xffffu), bf16_val(q.x >> 16), bf16_val(q.y & 0xffffu), bf16_val(q.y >> 16),
                               bf16_val(q.z & 0xffffu), bf16_val(q.z >> 16), bf16_val(q.w & 0xffffu), bf16_val(q.w >> 16)};
                for (int t = 0; t < tn; t++) {
                    const float* xr = x + (long)(t0 + t) * xs + xoff + c;
                    float4 a = *(const float4*)xr, b = *(const float4*)(xr + 4);
                    acc[t] += a.x * wf[0];
                    acc[t] += a.y * wf[1];
                    acc[t] += a.z * wf[2];
                    acc[t] += a.w * wf[3];
                    acc[t] += b.x * wf[4];
                    acc[t] += b.y * wf[5];
                    acc[t] += b.z * wf[6];
                    acc[t] += b.w * wf[7];
                }
            }
        } else {
            for (int c = lane; c < k; c += 32) {
                float wv = bf16_val(wr[c]);
                for (int t = 0; t < tn; t++) acc[t] += x[(long)(t0 + t) * xs + xoff + c] * wv;
            }
        }
        for (int t = 0; t < tn; t++) {
            float v = warp_sum(acc[t]);
            if (lane == 0) y[(long)(t0 + t) * n + row] = round ? to_bf16(v) : v;
        }
    }
}

// gemv_bf16 on fp8 weights (32x32-tile e8m0 scales) that were never
// expanded: each weight is fp8 * scale, the exact value the bf16 copy held,
// and the lanes walk the row exactly as gemv_bf16 does (8 consecutive weights
// per step), so the results are the bf16 kernel's, from half the bytes. The
// activation is not quantized (the reference runs wo_a in bf16).
__global__ void gemv_fp8w(const float* x, const unsigned char* w, const unsigned char* s, float* y, int n, int k, int nt,
                          int round, int group_rows, int x_stride) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    const unsigned char* wr = w + (long)row * k;
    int kb = (k + 31) / 32;
    const unsigned char* sr = s + (long)(row / 32) * kb;
    int xoff = group_rows > 0 ? (row / group_rows) * k : 0;
    int xs = group_rows > 0 ? x_stride : k;
    bool vec = (k % 8 == 0) && ((((unsigned long long)wr) & 7ull) == 0ull) && aligned16(x) && (xs % 4 == 0) && (xoff % 4 == 0);
    for (int t0 = 0; t0 < nt; t0 += GEMV_TOK) {
        float acc[GEMV_TOK] = {0};
        int tn = min(GEMV_TOK, nt - t0);
        if (vec) {
            for (int c = lane * 8; c < k; c += 256) {
                uint2 q = *(const uint2*)(wr + c);
                float sc = e8m0_val(sr[c / 32]);
                float wf[8];
                fp8x2(q.x, wf[0], wf[1]);
                fp8x2(q.x >> 16, wf[2], wf[3]);
                fp8x2(q.y, wf[4], wf[5]);
                fp8x2(q.y >> 16, wf[6], wf[7]);
#pragma unroll
                for (int i = 0; i < 8; i++) wf[i] *= sc;
                for (int t = 0; t < tn; t++) {
                    const float* xr = x + (long)(t0 + t) * xs + xoff + c;
                    float4 a = *(const float4*)xr, b = *(const float4*)(xr + 4);
                    acc[t] += a.x * wf[0];
                    acc[t] += a.y * wf[1];
                    acc[t] += a.z * wf[2];
                    acc[t] += a.w * wf[3];
                    acc[t] += b.x * wf[4];
                    acc[t] += b.y * wf[5];
                    acc[t] += b.z * wf[6];
                    acc[t] += b.w * wf[7];
                }
            }
        } else {
            for (int c = lane; c < k; c += 32) {
                float wv = fp8_val(wr[c]) * e8m0_val(sr[c / 32]);
                for (int t = 0; t < tn; t++) acc[t] += x[(long)(t0 + t) * xs + xoff + c] * wv;
            }
        }
        for (int t = 0; t < tn; t++) {
            float v = warp_sum(acc[t]);
            if (lane == 0) y[(long)(t0 + t) * n + row] = round ? to_bf16(v) : v;
        }
    }
}

__global__ void gemv_f32(const float* x, const float* w, float* y, int n, int k, int nt) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    const float* wr = w + (long)row * k;
    for (int t0 = 0; t0 < nt; t0 += GEMV_TOK) {
        float acc[GEMV_TOK] = {0};
        int tn = min(GEMV_TOK, nt - t0);
        for (int c = lane; c < k; c += 32) {
            float wv = wr[c];
            for (int t = 0; t < tn; t++) acc[t] += x[(long)(t0 + t) * k + c] * wv;
        }
        for (int t = 0; t < tn; t++) {
            float v = warp_sum(acc[t]);
            if (lane == 0) y[(long)(t0 + t) * n + row] = v;
        }
    }
}

// Packed e2m1 weights [n][k/2] (low nibble first) with one e8m0 scale per 32
// along k; x already fp8-quantized. Per 32-block partial sum, then scale.
__global__ void gemv_fp4(const float* x, const unsigned char* w, const unsigned char* s, float* y,
                         int n, int k, int nt, int round) {
    __shared__ float lut[16];  // per-lane nibbles would serialize on the constant cache
    if (threadIdx.x < 16) lut[threadIdx.x] = FP4[threadIdx.x];
    __syncthreads();
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    int kb = k / 32;
    const unsigned char* wr = w + (long)row * (k / 2);
    const unsigned char* sr = s + (long)row * kb;
    bool wv16 = aligned16(wr), xv = aligned16(x);
    for (int t0 = 0; t0 < nt; t0 += GEMV_TOK) {
        float acc[GEMV_TOK] = {0};
        int tn = min(GEMV_TOK, nt - t0);
        for (int b = lane; b < kb; b += 32) {
            float sc = e8m0_val(sr[b]);
            float part[GEMV_TOK] = {0};
            uint4 q = wv16 ? *(const uint4*)(wr + b * 16) : make_uint4(0, 0, 0, 0);
            if (!wv16) {
                unsigned int v[4] = {0, 0, 0, 0};
                for (int i = 0; i < 16; i++) v[i >> 2] |= (unsigned int)wr[b * 16 + i] << (8 * (i & 3));
                q = make_uint4(v[0], v[1], v[2], v[3]);
            }
            for (int t = 0; t < tn; t++) {
                float xf[32];
                load32(x + (long)(t0 + t) * k + b * 32, xv, xf);
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    unsigned int byte = byte_of(q, i);
                    part[t] += xf[2 * i] * lut[byte & 0xf] + xf[2 * i + 1] * lut[byte >> 4];
                }
            }
            for (int t = 0; t < tn; t++) acc[t] += part[t] * sc;
        }
        for (int t = 0; t < tn; t++) {
            float v = warp_sum(acc[t]);
            if (lane == 0) y[(long)(t0 + t) * n + row] = round ? to_bf16(v) : v;
        }
    }
}

// ---------------------------------------------------------------- grouped experts (decode)

// Routed experts for ONE token, all selected experts in one launch each
// stage, described by one table of 3 * nexp u64: tab[s] is the device address
// of expert s's record (w1|w2|w3 packed e2m1 then s1|s2|s3 e8m0 scales, the
// dsv41::expert layout), tab[nexp + s] the output row it writes, and
// tab[2 * nexp + s] the bits of its f32 route weight. Per element the
// arithmetic is gemv_fp4 + swiglu, unchanged.

// h[s][r] = bf16(silu(min(bf16(g), lim)) * clamp(bf16(u), lim) * route[s]),
// g/u = row r of w1/w3 against the fp8-quantized xq. One warp per (s, r).
__global__ void moe_gate_up(const float* xq, const unsigned long long* recs, float* h,
                            int nexp, int inter, int dim, float lim) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = FP4[threadIdx.x];
    __syncthreads();
    int item = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (item >= nexp * inter) return;
    int s = item / inter, r = item % inter;
    const unsigned char* rec = (const unsigned char*)recs[s];
    long W = (long)inter * dim / 2, S = (long)inter * dim / 32;
    const unsigned char* w1 = rec + (long)r * (dim / 2);
    const unsigned char* w3 = rec + 2 * W + (long)r * (dim / 2);
    const unsigned char* s1 = rec + 3 * W + (long)r * (dim / 32);
    const unsigned char* s3 = rec + 3 * W + 2 * S + (long)r * (dim / 32);
    int kb = dim / 32;
    bool xv = aligned16(xq);
    float ag = 0.0f, au = 0.0f;
    // records are 16-byte aligned slots and rows are dim/2 bytes (dim % 32 == 0)
#pragma unroll 2
    for (int b = lane; b < kb; b += 32) {
        uint4 qg = *(const uint4*)(w1 + b * 16), qu = *(const uint4*)(w3 + b * 16);
        float xf[32];
        load32(xq + b * 32, xv, xf);
        float pg = 0.0f, pu = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned int bg = byte_of(qg, i), bu = byte_of(qu, i);
            pg += xf[2 * i] * lut[bg & 0xf] + xf[2 * i + 1] * lut[bg >> 4];
            pu += xf[2 * i] * lut[bu & 0xf] + xf[2 * i + 1] * lut[bu >> 4];
        }
        ag += pg * e8m0_val(s1[b]);
        au += pu * e8m0_val(s3[b]);
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) {
        float g = to_bf16(ag), u = to_bf16(au);
        if (lim > 0.0f) {
            u = fminf(fmaxf(u, -lim), lim);
            g = fminf(g, lim);
        }
        float w = __uint_as_float((unsigned int)recs[2 * nexp + s]);
        h[(long)s * inter + r] = to_bf16(g / (1.0f + expf(-g)) * u * w);
    }
}

// out[row(s)][r] = bf16(row r of w2 . hq[s]), hq already fp8-quantized. One warp per (s, r).
__global__ void moe_down(const float* hq, const unsigned long long* recs, float* out, int nexp, int inter, int dim) {
    // a half-warp per output row (two rows per warp): rows are only
    // inter / 32 = 72 blocks long, which 32 lanes would split unevenly
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = FP4[threadIdx.x];
    __syncthreads();
    int item = (blockIdx.x * blockDim.x + threadIdx.x) / 16;
    int hl = threadIdx.x & 15;
    bool on = item < nexp * dim;
    int s = on ? item / dim : 0, r = on ? item % dim : 0;
    const unsigned char* rec = (const unsigned char*)recs[s];
    long W = (long)inter * dim / 2, S = (long)inter * dim / 32;
    const unsigned char* w2 = rec + W + (long)r * (inter / 2);
    const unsigned char* s2 = rec + 3 * W + S + (long)r * (inter / 32);
    const float* x = hq + (long)s * inter;
    int kb = inter / 32;
    bool xv = aligned16(x);
    float acc = 0.0f;
    if (on) {
#pragma unroll 2
        for (int b = hl; b < kb; b += 16) {
            uint4 q = *(const uint4*)(w2 + b * 16);
            float xf[32];
            load32(x + b * 32, xv, xf);
            float part = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                unsigned int byte = byte_of(q, i);
                part += xf[2 * i] * lut[byte & 0xf] + xf[2 * i + 1] * lut[byte >> 4];
            }
            acc += part * e8m0_val(s2[b]);
        }
    }
    for (int o = 8; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (on && hl == 0) out[(long)recs[nexp + s] * dim + r] = to_bf16(acc);
}

// y[k] = bf16((sum_s out[s][k]) + shared[k]): the f32 accumulation in slot
// order (the caller orders slots by expert id, as the reference loop does),
// then the shared expert, then one bf16 round.
__global__ void moe_reduce(const float* out, const float* shared, float* y, int nexp, int dim) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= dim) return;
    float acc = 0.0f;
    for (int s = 0; s < nexp; s++) acc += out[(long)s * dim + k];
    y[k] = to_bf16(acc + shared[k]);
}

// moe_reduce where rows set in `host_mask` come from the CPU, in pinned
// host memory (handoff.rs): each block first waits until the CPU job has
// published `seq`, then sums rows in the same order as moe_reduce.
__global__ void moe_reduce_host(const float* out, const float* shared, float* y, int nexp, int dim,
                                const float* host_rows, unsigned int host_mask, const unsigned int* flag,
                                unsigned int seq) {
    if (threadIdx.x == 0) {
        while (*(volatile const unsigned int*)flag < seq) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(500);
#endif
        }
    }
    __syncthreads();
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= dim) return;
    float acc = 0.0f;
    for (int s = 0; s < nexp; s++)
        acc += ((host_mask >> s) & 1u) ? ((volatile const float*)host_rows)[(long)s * dim + k] : out[(long)s * dim + k];
    y[k] = to_bf16(acc + shared[k]);
}

// Copy a (na floats) then b (nb floats) into host-mapped memory `dst` and
// raise `flag` to `seq` once all of it is visible to the host (Inbox).
// One block.
__global__ void publish(const float* a, int na, const float* b, int nb, float* dst, unsigned int* flag, unsigned int seq) {
    for (int i = threadIdx.x; i < na; i += blockDim.x) dst[i] = a[i];
    for (int i = threadIdx.x; i < nb; i += blockDim.x) dst[na + i] = b[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile unsigned int*)flag = seq;
        __threadfence_system();
    }
}

// ---------------------------------------------------------------- elementwise

// SwiGLU with the training clamps; gate/up already bf16; optional per-token
// route weight (w == nullptr: none). h[t][i] = bf16(silu(g) * u * w[t]).
__global__ void swiglu(const float* gate, const float* up, const float* w, float* h, int inter, int nt, float lim) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)inter * nt) return;
    float g = gate[i], u = up[i];
    if (lim > 0.0f) {
        u = fminf(fmaxf(u, -lim), lim);
        g = fminf(g, lim);
    }
    float v = g / (1.0f + expf(-g)) * u;
    if (w) v *= w[i / inter];
    h[i] = to_bf16(v);
}

// dst[t][:] = src[idx[t]][:] (the tokens routed to one expert, as a batch)
__global__ void gather_rows(const float* src, const int* idx, float* dst, int d, int nt) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)d * nt) return;
    int t = (int)(i / d), c = (int)(i % d);
    dst[i] = src[(long)idx[t] * d + c];
}

// acc[idx[t]][:] += src[t][:] (expert outputs into the f32 MoE accumulator)
__global__ void scatter_add_rows(float* acc, const float* src, const int* idx, int d, int nt) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)d * nt) return;
    int t = (int)(i / d), c = (int)(i % d);
    acc[(long)idx[t] * d + c] += src[i];
}

// y = bf16(a + b)
__global__ void add_round(const float* a, const float* b, float* y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = to_bf16(a[i] + b[i]);
}

__global__ void round_bf16(float* x, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = to_bf16(x[i]);
}

// ---------------------------------------------------------------- norms, rope

// ops::rmsnorm: y = bf16(w * (x * (1 / sqrt(mean(x^2) + eps)))), one block per row
// Sum over the block (blockDim.x a multiple of 32, at most 1024) via warp
// shuffles and 32 floats of shared memory; every thread gets the total.
__device__ __forceinline__ float block_sum(float v, float* red) {
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32, nw = blockDim.x / 32;
    v = warp_sum(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : 0.0f;
    t = warp_sum(t);
    __syncthreads();  // red may be reused by the caller
    return t;
}

__global__ void rmsnorm(const float* x, const float* w, float* y, int d, float eps) {
    __shared__ float red[32];
    const float* xr = x + (long)blockIdx.x * d;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) ss += xr[c] * xr[c];
    float r = 1.0f / sqrtf(block_sum(ss, red) / (float)d + eps);
    for (int c = threadIdx.x; c < d; c += blockDim.x) y[(long)blockIdx.x * d + c] = to_bf16(w[c] * (xr[c] * r));
}

// ops::Rope::apply on rows: the pairs at x[row*stride + off .. + 2*half] rotated
// to positions pos[row] (inverse: conjugate), results rounded to bf16.
__global__ void rope(float* x, const float* cosv, const float* sinv, const int* pos, int rows, int stride, int off,
                     int half, int inverse, int pos0, int per) {
    // row r's position: pos[r], or pos0 + r / per when pos is null (decode's
    // rows share one position, so nothing needs uploading)
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)rows * half) return;
    int r = (int)(i / half), p = (int)(i % half);
    float* v = x + (long)r * stride + off + 2 * p;
    long pr = pos ? pos[r] : pos0 + r / per;
    float c = cosv[pr * half + p], s = sinv[pr * half + p];
    if (inverse) s = -s;
    float a = v[0], b = v[1];
    v[0] = to_bf16(a * c - b * s);
    v[1] = to_bf16(a * s + b * c);
}

// ---------------------------------------------------------------- hyper-connections

// hc::mixes, the part worth a GPU: per token, the 24 projections of its
// 4*d stream and the stream's sum of squares -> out[t][0..24], out[t][24].
// One block per (token, output j): the 25 sums run side by side (one block
// would do them one after another), each in the same order as ever.
__global__ void hc_project(const float* x, const float* fn, float* out, int n) {
    extern __shared__ float red[];
    const float* xr = x + (long)blockIdx.x * n;
    int j = blockIdx.y;
    float acc = 0.0f;
    const float* a = j < 24 ? fn + (long)j * n : xr;
    if (n % 4 == 0 && aligned16(a) && aligned16(xr)) {
        for (int c = 4 * threadIdx.x; c < n; c += 4 * blockDim.x) {
            float4 u = *(const float4*)(a + c), v = *(const float4*)(xr + c);
            acc += u.x * v.x;
            acc += u.y * v.y;
            acc += u.z * v.z;
            acc += u.w * v.w;
        }
    } else {
        for (int c = threadIdx.x; c < n; c += blockDim.x) acc += a[c] * xr[c];
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[(long)blockIdx.x * 25 + j] = red[0];
}

__device__ __forceinline__ void mix_token(const float* p, const float* base, const float* scale, float* o, int n,
                                          float norm_eps, int iters, float hc_eps, int base_lane, bool on);

// hc_project and hc_mix in one launch: the (token, output) blocks of
// hc_project, and the last block of each token to finish (an atomic ticket,
// after a fence) computes that token's mix with hc_mix's code, reading the
// other blocks' projections from L2. counter[t] returns to 0 afterwards.
__global__ void hc_project_mix(const float* x, const float* fn, float* out, int n, const float* base, const float* scale,
                               float* mix, float norm_eps, int iters, float hc_eps, unsigned int* counter) {
    extern __shared__ float red[];
    __shared__ bool last;
    const float* xr = x + (long)blockIdx.x * n;
    int j = blockIdx.y;
    float acc = 0.0f;
    const float* a = j < 24 ? fn + (long)j * n : xr;
    if (n % 4 == 0 && aligned16(a) && aligned16(xr)) {
        for (int c = 4 * threadIdx.x; c < n; c += 4 * blockDim.x) {
            float4 u = *(const float4*)(a + c), v = *(const float4*)(xr + c);
            acc += u.x * v.x;
            acc += u.y * v.y;
            acc += u.z * v.z;
            acc += u.w * v.w;
        }
    } else {
        for (int c = threadIdx.x; c < n; c += blockDim.x) acc += a[c] * xr[c];
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[(long)blockIdx.x * 25 + j] = red[0];
        __threadfence();
        last = atomicAdd(counter + blockIdx.x, 1u) == 24u;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < 32) mix_token(out + (long)blockIdx.x * 25, base, scale, mix + (long)blockIdx.x * 24, n, norm_eps, iters,
                                    hc_eps, threadIdx.x & 16, threadIdx.x < 16);
    if (threadIdx.x == 0) counter[blockIdx.x] = 0u;
}

// hc::mixes_from_projection per token, on the device: proj[t] = {24 raw
// projections, sum of squares} (hc_project), n = the stream width. Writes
// mix[t] = {pre[4], post[4], comb[16] (row i = residual copy)}: the same
// operations in the same order as the CPU, so no host round trip is needed
// between the projection and the mixing. One thread per token.
// hc::mixes_from_projection for one token, on 16 lanes (lane e = comb
// entry (e / 4, e % 4)) of a warp; `base_lane` is 0 or 16 (which half of the
// warp), `on` whether to write. `p` = the token's 25 projection values.
__device__ __forceinline__ void mix_token(const float* p, const float* base, const float* scale, float* o, int n,
                                          float norm_eps, int iters, float hc_eps, int base_lane, bool on) {
    int e = threadIdx.x & 15, j = e >> 2, k = e & 3;
    float r = 1.0f / sqrtf(__ldcg(p + 24) / (float)n + norm_eps);
    if (on && e < 4) {
        o[e] = 1.0f / (1.0f + expf(-((__ldcg(p + e) * r) * scale[0] + base[e]))) + hc_eps;
        o[4 + e] = 2.0f * (1.0f / (1.0f + expf(-((__ldcg(p + 4 + e) * r) * scale[1] + base[4 + e]))));
    }
    float c = (__ldcg(p + 8 + e) * r) * scale[2] + base[8 + e];
#define ROW(q) __shfl_sync(0xffffffffu, c, base_lane + j * 4 + (q))
#define COL(q) __shfl_sync(0xffffffffu, c, base_lane + (q) * 4 + k)
    float mx = fmaxf(fmaxf(fmaxf(fmaxf(-INF, ROW(0)), ROW(1)), ROW(2)), ROW(3));
    c = expf(c - mx);
    float s = 0.0f;
    s += ROW(0); s += ROW(1); s += ROW(2); s += ROW(3);
    c = c / s + hc_eps;
    for (int it = 0; it < iters; it++) {
        if (it > 0) {
            float rs = 0.0f;
            rs += ROW(0); rs += ROW(1); rs += ROW(2); rs += ROW(3);
            rs += hc_eps;
            c /= rs;
        }
        float cs = 0.0f;
        cs += COL(0); cs += COL(1); cs += COL(2); cs += COL(3);
        cs += hc_eps;
        c /= cs;
    }
#undef ROW
#undef COL
    if (on) o[8 + e] = c;
}

__global__ void hc_mix(const float* proj, const float* base, const float* scale, float* mix, int nt, int n,
                       float norm_eps, int iters, float hc_eps) {
    // 16 lanes per token
    int g = (blockIdx.x * blockDim.x + threadIdx.x) / 16;
    bool on = g < nt;
    int tok = on ? g : 0;
    mix_token(proj + (long)tok * 25, base, scale, mix + (long)tok * 24, n, norm_eps, iters, hc_eps, threadIdx.x & 16, on);
}

// hc::pre: y[t][k] = bf16(sum_i pre[t][i] * x[t][i][k]); pre[t] starts at
// mix[t * stride] (stride 24 for an hc_mix buffer, 4 for bare pre vectors)
// hc_pre then rmsnorm in one launch, bit for bit: the collapsed copy goes
// to shared memory, and the sum of squares uses rmsnorm's 256 threads and
// reduction tree. One block of 256 per token; d + 256 floats of shared memory.
// Decode's new kv row, from the wkv output to its window slot in one launch:
// rmsnorm (rmsnorm's own reduction), rope on the last 2 * half values at
// position pos, act_quant_fp8, write. A thread per value (d <= 1024,
// d % 32 == 0), so the same bits as the four steps apart.
__global__ void kv_finish(const float* kv0, const float* w, const float* cosv, const float* sinv, int pos, int half,
                          float* dst, int d, float eps) {
    __shared__ float red[32];
    int c = threadIdx.x;
    float v = c < d ? kv0[c] : 0.0f;
    float ss = c < d ? v * v : 0.0f;
    float r = 1.0f / sqrtf(block_sum(ss, red) / (float)d + eps);
    float y = c < d ? to_bf16(w[c] * (v * r)) : 0.0f;
    int off = d - 2 * half;
    float other = __shfl_xor_sync(0xffffffffu, y, 1);
    if (c < d && c >= off) {
        int p = (c - off) / 2;
        float cs = cosv[(long)pos * half + p], sn = sinv[(long)pos * half + p];
        bool even = ((c - off) & 1) == 0;
        float a = even ? y : other, b = even ? other : y;
        y = even ? to_bf16(a * cs - b * sn) : to_bf16(a * sn + b * cs);
    }
    float amax = fmaxf(warp_max(fabsf(y)), 1e-4f);
    float sc = pow2_ceil(amax * (1.0f / 448.0f));
    if (c < d) dst[c] = round_e4m3(fminf(fmaxf(y / sc, -448.0f), 448.0f)) * sc;
}

__global__ void hc_pre_norm(const float* x, const float* mix, int stride, const float* w, float* y, int d, float eps) {
    // hc_pre then rmsnorm in one launch (rmsnorm's reduction, so the same
    // bits as the two launches): the collapsed copy stays in shared memory
    extern __shared__ float xs[];
    __shared__ float red[32];
    int t = blockIdx.x;
    const float* xt = x + (long)t * 4 * d;
    const float* p = mix + (long)t * stride;
    float ss = 0.0f;
    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < 4; c++) acc += p[c] * xt[(long)c * d + k];
        float v = to_bf16(acc);
        xs[k] = v;
        ss += v * v;
    }
    float r = 1.0f / sqrtf(block_sum(ss, red) / (float)d + eps);
    for (int k = threadIdx.x; k < d; k += blockDim.x) y[(long)t * d + k] = to_bf16(w[k] * (xs[k] * r));
}

__global__ void hc_pre(const float* x, const float* mix, float* y, int d, int nt, int stride) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)d * nt) return;
    int t = (int)(i / d), k = (int)(i % d);
    const float* xt = x + (long)t * 4 * d;
    const float* p = mix + (long)t * stride;
    float acc = 0.0f;
    for (int c = 0; c < 4; c++) acc += p[c] * xt[(long)c * d + k];
    y[i] = to_bf16(acc);
}

// hc::post: y[t][j][k] = bf16(post[j] * out[t][k] + sum_i comb[i][j] * res[t][i][k]);
// mix[t] = {pre[4], post[4], comb[16]} as written by hc_mix
__global__ void hc_post(const float* out, const float* res, const float* mix, float* y, int d, int nt) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)4 * d * nt) return;
    int t = (int)(i / (4L * d)), j = (int)((i / d) % 4), k = (int)(i % d);
    const float* m = mix + (long)t * 24;
    const float* rt = res + (long)t * 4 * d;
    float r = 0.0f;
    for (int c = 0; c < 4; c++) r += m[8 + c * 4 + j] * rt[(long)c * d + k];
    y[i] = to_bf16(m[4 + j] * out[(long)t * d + k] + r);
}

// ---------------------------------------------------------------- attention

// attention::sparse_attend for every (token, head): block = (t, h). idx is
// [nt][nidx] (-1 = none); kv rows are hd wide. scores in dynamic shared memory.
__global__ void sparse_attn(const float* q, const float* kva, int rows_a, const float* kvb, const int* idx,
                            const float* sink, float* out, int nh, int hd, int nidx, float scale,
                            const float* cosv, const float* sinv, int rope_off, int rope_half, int pos0) {
    // cosv non-null: the output's last 2 * rope_half values get the inverse
    // rotation of rope() (position pos0 + token), adjacent columns swapping
    // values by shuffle; needs a thread per column (blockDim.x == hd).
    // kv row p is kva[p] below rows_a, else kvb[p - rows_a]: the window and
    // the compressed cache are read where they live, no concatenated copy
#define KVROW(p) ((p) < rows_a ? kva + (long)(p) * hd : kvb + (long)((p) - rows_a) * hd)
    // shared: sc[nidx] scores then weights, red[32] for the max, denom
    extern __shared__ float sc[];
    float* red = sc + nidx;
    int t = blockIdx.x / nh, h = blockIdx.x % nh;
    const float* qh = q + ((long)t * nh + h) * hd;
    const int* it = idx + (long)t * nidx;
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31, nw = blockDim.x / 32;
    for (int j = warp; j < nidx; j += nw) {
        int p = it[j];
        float acc = 0.0f;
        if (p >= 0) {
            const float* kr = KVROW(p);
            for (int c = lane; c < hd; c += 32) acc += qh[c] * kr[c];
        }
        acc = warp_sum(acc);
        if (lane == 0) sc[j] = p >= 0 ? acc * scale : -INF;
    }
    __syncthreads();
    // max: order-free, so a parallel reduction gives the sequential result
    float m = -1e30f;
    for (int j = threadIdx.x; j < nidx; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = warp_max(m);
    if (lane == 0) red[warp] = m;
    __syncthreads();
    float mx = -1e30f;
    for (int w = 0; w < nw; w++) mx = fmaxf(mx, red[w]);
    for (int j = threadIdx.x; j < nidx; j += blockDim.x) sc[j] = it[j] >= 0 ? expf(sc[j] - mx) : 0.0f;
    __syncthreads();
    // the denominator in the reference's order, by one thread (~1 us)
    if (threadIdx.x == 0) {
        float denom = 0.0f;
        for (int j = 0; j < nidx; j++) denom += sc[j];
        red[32] = denom + expf(sink[h] - mx);
    }
    __syncthreads();
    float denom = red[32];
    float* oh = out + ((long)t * nh + h) * hd;
    for (int c = threadIdx.x; c < hd; c += blockDim.x) {
        float acc = 0.0f;
#pragma unroll 4
        for (int j = 0; j < nidx; j++) {
            int p = it[j];
            if (p >= 0) acc += to_bf16(sc[j]) * KVROW(p)[c];
        }
        float v = to_bf16(acc / denom);
        if (cosv) {
            float other = __shfl_xor_sync(0xffffffffu, v, 1);
            if (c >= rope_off) {
                int p = (c - rope_off) / 2;
                long pr = pos0 + t;
                float cs = cosv[pr * rope_half + p], sn = -sinv[pr * rope_half + p];
                bool even = ((c - rope_off) & 1) == 0;
                float a = even ? v : other, b = even ? other : v;
                v = even ? to_bf16(a * cs - b * sn) : to_bf16(a * sn + b * cs);
            }
        }
        oh[c] = v;
    }
#undef KVROW
}

// Indexer scores: s[t][j] = bf16(sum_h bf16(relu(bf16(q[t][h] . k[j])) * w[t][h]))
__global__ void index_scores(const float* q, const float* k, const float* w, float* s, int nh, int ihd, int nkeys,
                             int nt, float wscale) {
    // w holds the raw weights_proj output; the head weight is bf16(w * wscale),
    // as the host used to compute it
    // one block per (token, key), one thread per head: each head's dot in
    // the same order as before, then thread 0 adds the heads in order
    extern __shared__ float term[];
    int t = blockIdx.x / nkeys, j = blockIdx.x % nkeys, h = threadIdx.x;
    if (h < nh) {
        const float* kj = k + (long)j * ihd;
        const float* qh = q + ((long)t * nh + h) * ihd;
        float dot = 0.0f;
        for (int c = 0; c < ihd; c++) dot += qh[c] * kj[c];
        float d = fmaxf(to_bf16(dot), 0.0f);
        term[h] = to_bf16(d * to_bf16(w[t * nh + h] * wscale));
    }
    __syncthreads();
    if (h == 0) {
        float acc = 0.0f;
        for (int i = 0; i < nh; i++) acc += term[i];
        s[(long)t * nkeys + j] = to_bf16(acc);
    }
}

// Compressor pooling: out[g][f] = sum_i kv[g*r+i][f] * softmax_i(score[g*r+i][f])
__global__ void compress_pool(const float* kv, const float* score, float* out, int groups, int r, int hd) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)groups * hd) return;
    int g = (int)(i / hd), f = (int)(i % hd);
    float mx = -INF;
    for (int j = 0; j < r; j++) mx = fmaxf(mx, score[((long)g * r + j) * hd + f]);
    float e[8], s = 0.0f;
    for (int j = 0; j < r; j++) {
        e[j] = expf(score[((long)g * r + j) * hd + f] - mx);
        s += e[j];
    }
    float acc = 0.0f;
    for (int j = 0; j < r; j++) acc += kv[((long)g * r + j) * hd + f] * (e[j] / s);
    out[i] = to_bf16(acc);
}

// ---------------------------------------------------------------- engram

// engram::Engram gate for block (t, copy c): out = bf16(h + gate * value).
// kv[t] = {key[4][d], value[d]}; qk = q_weight * k_weight, [4][d].
__global__ void engram_gate(const float* h, const float* kv, const float* qk, float* out, int d, float eps, float scale) {
    extern __shared__ float red[];  // 3 * blockDim.x
    int t = blockIdx.x / 4, c = blockIdx.x % 4;
    const float* hs = h + ((long)t * 4 + c) * d;
    const float* key = kv + (long)t * 5 * d + (long)c * d;
    const float* value = kv + (long)t * 5 * d + 4L * d;
    const float* w = qk + (long)c * d;
    float a = 0.0f, b = 0.0f, dot = 0.0f;
    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        a += hs[k] * hs[k];
        b += key[k] * key[k];
        dot += hs[k] * w[k] * key[k];
    }
    float* ra = red;
    float* rb = red + blockDim.x;
    float* rd = red + 2 * blockDim.x;
    ra[threadIdx.x] = a;
    rb[threadIdx.x] = b;
    rd[threadIdx.x] = dot;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            ra[threadIdx.x] += ra[threadIdx.x + o];
            rb[threadIdx.x] += rb[threadIdx.x + o];
            rd[threadIdx.x] += rd[threadIdx.x + o];
        }
        __syncthreads();
    }
    float rstd = 1.0f / sqrtf(ra[0] / (float)d + eps) * (1.0f / sqrtf(rb[0] / (float)d + eps));
    float dt = rd[0] * rstd * scale;
    float g = 1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(dt), 1e-6f)), dt)));
    float* o = out + ((long)t * 4 + c) * d;
    for (int k = threadIdx.x; k < d; k += blockDim.x) o[k] = to_bf16(hs[k] + g * value[k]);
}

// ---------------------------------------------------------------- vision tower

// The ViT and aligner (dsv41::vision). No bit-exact target here: the
// reference runs cuBLAS bf16 GEMMs and fused attention, so these kernels use
// fused multiply-adds (fmaf, which --fmad=false leaves alone) and match the
// CPU tower to tolerance.

// y[t][n] = bf16?(x[t][k] . w[n][k] + bias[n]), w in bf16 bits: a bf16
// nn.Linear. A 64x64 output tile per 256-thread block, 4x4 outputs a thread,
// k in steps of 16 through shared memory.
__global__ void gemm_bf16(const float* x, const unsigned short* w, const float* bias, float* y, int t, int n, int k, int round) {
    __shared__ __align__(16) float xs[16][68];
    __shared__ __align__(16) float ws[16][68];
    int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < k; k0 += 16) {
        for (int e = threadIdx.x; e < 1024; e += 256) {
            int r = e >> 4, c = e & 15, gc = k0 + c;
            int xr = m0 + r, wr = n0 + r;
            xs[c][r] = (xr < t && gc < k) ? x[(long)xr * k + gc] : 0.0f;
            ws[c][r] = (wr < n && gc < k) ? bf16_val(w[(long)wr * k + gc]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            float4 a = *(const float4*)&xs[kk][ty * 4];
            float4 b = *(const float4*)&ws[kk][tx * 4];
            float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = m0 + ty * 4 + i;
        if (r >= t) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int c = n0 + tx * 4 + j;
            if (c >= n) continue;
            float v = acc[i][j] + bias[c];
            y[(long)r * n + c] = round ? to_bf16(v) : v;
        }
    }
}

// vision::attention's rotation: q and k of qkv [n][3][heads][hd] turned in
// place by the 2-D tables [n][hd/2] (pairs c, c + hd/2), rounded to bf16.
__global__ void vit_rope(float* qkv, const float* cosv, const float* sinv, int n, int heads, int hd) {
    int half = hd / 2;
    long total = (long)n * 2 * heads * half;
    long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= total) return;
    int c = (int)(e % half);
    long rest = e / half;
    int h = (int)(rest % heads);
    rest /= heads;
    int which = (int)(rest % 2);
    long i = rest / 2;
    float* x = qkv + i * 3 * heads * hd + (long)which * heads * hd + (long)h * hd;
    float cs = cosv[i * half + c], sn = sinv[i * half + c];
    float x1 = x[c], x2 = x[c + half];
    x[c] = to_bf16(x1 * cs - x2 * sn);
    x[c + half] = to_bf16(x2 * cs + x1 * sn);
}

// Full (bidirectional) attention over one image, head size 64: qkv
// [n][3][heads][64] (q, k rotated), out [n][heads * 64] in bf16. One thread
// per query, 128 queries a block, keys through shared memory 32 at a time,
// online softmax in f32.
__global__ void vit_attn(const float* qkv, float* out, int n, int heads, float scale) {
    __shared__ float ks[32][64];
    __shared__ float vs[32][64];
    int h = blockIdx.y, d = heads * 64;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool live = i < n;
    float q[64], o[64];
#pragma unroll
    for (int c = 0; c < 64; c++) {
        q[c] = live ? qkv[(long)i * 3 * d + h * 64 + c] * scale : 0.0f;
        o[c] = 0.0f;
    }
    float m = -INF, l = 0.0f;
    for (int j0 = 0; j0 < n; j0 += 32) {
        __syncthreads();
        for (int e = threadIdx.x; e < 32 * 64; e += blockDim.x) {
            int r = e >> 6, c = e & 63, j = j0 + r;
            ks[r][c] = j < n ? qkv[(long)j * 3 * d + d + h * 64 + c] : 0.0f;
            vs[r][c] = j < n ? qkv[(long)j * 3 * d + 2 * d + h * 64 + c] : 0.0f;
        }
        __syncthreads();
        int nj = min(32, n - j0);
        float s[32];
        float mt = m;
#pragma unroll
        for (int r = 0; r < 32; r++) {
            float acc = 0.0f;
#pragma unroll
            for (int c = 0; c < 64; c++) acc = fmaf(q[c], ks[r][c], acc);
            s[r] = r < nj ? acc : -INF;
            mt = fmaxf(mt, s[r]);
        }
        float corr = expf(m - mt);
        l *= corr;
#pragma unroll
        for (int c = 0; c < 64; c++) o[c] *= corr;
#pragma unroll
        for (int r = 0; r < 32; r++) {
            float p = expf(s[r] - mt);
            l += p;
#pragma unroll
            for (int c = 0; c < 64; c++) o[c] = fmaf(p, vs[r][c], o[c]);
        }
        m = mt;
    }
    if (live) {
#pragma unroll
        for (int c = 0; c < 64; c++) out[(long)i * d + h * 64 + c] = to_bf16(o[c] / l);
    }
}

// The ViT MLP's gate: m[i][j] = bf16(bf16(silu(g)) * u), g = gu[i][j],
// u = gu[i][inter + j].
__global__ void silu_mul(const float* gu, float* m, int inter, long total) {
    long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= total) return;
    long i = e / inter;
    int j = (int)(e % inter);
    float g = gu[i * 2 * inter + j], u = gu[i * 2 * inter + inter + j];
    m[e] = to_bf16(to_bf16(g / (1.0f + expf(-g))) * u);
}

// x = bf16(gelu(x)), the exact (erf) GELU.
__global__ void gelu_round(float* x, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = to_bf16(v * 0.5f * (1.0f + erff(v * 0.70710678118654752f)));
    }
}

}  // extern "C"
