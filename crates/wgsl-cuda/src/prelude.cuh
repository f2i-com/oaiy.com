// wgsl-cuda's prelude: WGSL's types and built-ins for a kernel translated to CUDA C++ (crate wgsl-cuda).
//
// A WGSL vector is `vec<T, N>`, laid out as WGSL lays it out in memory (a vec2 aligned to two scalars, a vec3 and a vec4
// to four, a vec3 12 bytes and 16 apart in an array), its components `c[i]`. Its operators are WGSL's: component-wise,
// with a scalar on either side, comparisons a vector of bool, shifts by the amount modulo the width.
#include <cuda_fp16.h>

typedef unsigned int uint;
typedef unsigned long long u64;

template <typename T, int N> struct vec_align { static constexpr int value = sizeof(T) * (N == 3 ? 4 : N); };
template <typename T, int N> struct alignas(vec_align<T, N>::value) vec {
    T c[N];
    vec() = default;
    __device__ __forceinline__ explicit vec(T s) {
#pragma unroll
        for (int i = 0; i < N; i++) c[i] = s;
    }
    template <typename... A, typename = typename std::enable_if<sizeof...(A) == N && (N > 1)>::type>
    __device__ __forceinline__ vec(A... a) : c{T(a)...} {}
    __device__ __forceinline__ T& operator[](uint i) { return c[i]; }
    __device__ __forceinline__ const T& operator[](uint i) const { return c[i]; }
};

// a struct's vec3 member: its 12 bytes, as WGSL lays a struct out (the member after it may take the next 4, which a
// vec<T, 3> of 16 would cover); a vector as it is read, and written from one
template <typename T> struct pvec3 {
    T c[3];
    pvec3() = default;
    __device__ __forceinline__ pvec3(const vec<T, 3>& v) : c{v.c[0], v.c[1], v.c[2]} {}
    __device__ __forceinline__ pvec3& operator=(const vec<T, 3>& v) {
        c[0] = v.c[0];
        c[1] = v.c[1];
        c[2] = v.c[2];
        return *this;
    }
    __device__ __forceinline__ T& operator[](uint i) { return c[i]; }
    __device__ __forceinline__ const T& operator[](uint i) const { return c[i]; }
};
template <typename T> __device__ __forceinline__ vec<T, 3> wgsl_v3(const pvec3<T>& p) { return vec<T, 3>(p.c[0], p.c[1], p.c[2]); }
template <typename T> __device__ __forceinline__ vec<T, 3> wgsl_v3(const vec<T, 3>& v) { return v; }

// component-wise maps, one and two operands
template <typename R, typename T, int N, typename F> __device__ __forceinline__ vec<R, N> vmap(const vec<T, N>& a, F f) {
    vec<R, N> r;
#pragma unroll
    for (int i = 0; i < N; i++) r.c[i] = f(a.c[i]);
    return r;
}
template <typename R, typename T, typename U, int N, typename F>
__device__ __forceinline__ vec<R, N> vmap2(const vec<T, N>& a, const vec<U, N>& b, F f) {
    vec<R, N> r;
#pragma unroll
    for (int i = 0; i < N; i++) r.c[i] = f(a.c[i], b.c[i]);
    return r;
}

template <typename T> struct shift_mask { static constexpr uint value = sizeof(T) * 8 - 1; };

#define WGSL_ARITH(OP)                                                                                                       \
    template <typename T, int N> __device__ __forceinline__ vec<T, N> operator OP(const vec<T, N>& a, const vec<T, N>& b) { \
        return vmap2<T>(a, b, [](T x, T y) { return T(x OP y); });                                                       \
    }                                                                                                                        \
    template <typename T, int N> __device__ __forceinline__ vec<T, N> operator OP(const vec<T, N>& a, T b) {               \
        return vmap<T>(a, [b](T x) { return T(x OP b); });                                                               \
    }                                                                                                                        \
    template <typename T, int N> __device__ __forceinline__ vec<T, N> operator OP(T a, const vec<T, N>& b) {               \
        return vmap<T>(b, [a](T y) { return T(a OP y); });                                                               \
    }
WGSL_ARITH(+)
WGSL_ARITH(-)
WGSL_ARITH(*)
WGSL_ARITH(/)
WGSL_ARITH(&)
WGSL_ARITH(|)
WGSL_ARITH(^)
#undef WGSL_ARITH

#define WGSL_COMPARE(OP)                                                                                                        \
    template <typename T, int N> __device__ __forceinline__ vec<bool, N> operator OP(const vec<T, N>& a, const vec<T, N>& b) { \
        return vmap2<bool>(a, b, [](T x, T y) { return x OP y; });                                                          \
    }
WGSL_COMPARE(==)
WGSL_COMPARE(!=)
WGSL_COMPARE(<)
WGSL_COMPARE(<=)
WGSL_COMPARE(>)
WGSL_COMPARE(>=)
#undef WGSL_COMPARE

template <typename T, int N> __device__ __forceinline__ vec<T, N> operator-(const vec<T, N>& a) { return vmap<T>(a, [](T x) { return T(-x); }); }
template <typename T, int N> __device__ __forceinline__ vec<T, N> operator~(const vec<T, N>& a) { return vmap<T>(a, [](T x) { return T(~x); }); }
template <int N> __device__ __forceinline__ vec<bool, N> operator!(const vec<bool, N>& a) { return vmap<bool>(a, [](bool x) { return !x; }); }

// WGSL's integer / and %, which never fault: by zero the dividend and 0, the most negative by -1 itself and 0
__device__ __forceinline__ uint wgsl_div(uint a, uint b) { return a / (b == 0u ? 1u : b); }
__device__ __forceinline__ int wgsl_div(int a, int b) { return a / ((b == 0 || (a == int(0x80000000) && b == -1)) ? 1 : b); }
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_div(const vec<T, N>& a, const vec<T, N>& b) { return vmap2<T>(a, b, [](T x, T y) { return wgsl_div(x, y); }); }
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_div(const vec<T, N>& a, T b) { return vmap<T>(a, [b](T x) { return wgsl_div(x, b); }); }
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_div(T a, const vec<T, N>& b) { return vmap<T>(b, [a](T y) { return wgsl_div(a, y); }); }

// WGSL's %: an integer's remainder (as its division, above); a float's truncated one
__device__ __forceinline__ uint wgsl_rem(uint a, uint b) { return a % (b == 0u ? 1u : b); }
__device__ __forceinline__ int wgsl_rem(int a, int b) { return a % ((b == 0 || (a == int(0x80000000) && b == -1)) ? 1 : b); }
__device__ __forceinline__ float wgsl_rem(float a, float b) { return fmodf(a, b); }
__device__ __forceinline__ __half wgsl_rem(__half a, __half b) { return __float2half(fmodf(__half2float(a), __half2float(b))); }
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_rem(const vec<T, N>& a, const vec<T, N>& b) { return vmap2<T>(a, b, [](T x, T y) { return wgsl_rem(x, y); }); }
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_rem(const vec<T, N>& a, T b) { return vmap<T>(a, [b](T x) { return wgsl_rem(x, b); }); }
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_rem(T a, const vec<T, N>& b) { return vmap<T>(b, [a](T y) { return wgsl_rem(a, y); }); }

// WGSL's shifts: by the amount modulo the width
template <typename T> __device__ __forceinline__ T wgsl_shl(T a, uint b) { return T(a << (b & shift_mask<T>::value)); }
template <typename T> __device__ __forceinline__ T wgsl_shr(T a, uint b) { return T(a >> (b & shift_mask<T>::value)); }
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_shl(const vec<T, N>& a, const vec<uint, N>& b) { return vmap2<T>(a, b, [](T x, uint y) { return wgsl_shl(x, y); }); }
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_shr(const vec<T, N>& a, const vec<uint, N>& b) { return vmap2<T>(a, b, [](T x, uint y) { return wgsl_shr(x, y); }); }
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_shl(const vec<T, N>& a, uint b) { return vmap<T>(a, [b](T x) { return wgsl_shl(x, b); }); }
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_shr(const vec<T, N>& a, uint b) { return vmap<T>(a, [b](T x) { return wgsl_shr(x, b); }); }

// select(f, t, c): naga's Select { condition, accept, reject }
template <typename T> __device__ __forceinline__ T wgsl_select(bool c, T t, T f) { return c ? t : f; }
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_select(const vec<bool, N>& c, const vec<T, N>& t, const vec<T, N>& f) {
    vec<T, N> r;
#pragma unroll
    for (int i = 0; i < N; i++) r.c[i] = c.c[i] ? t.c[i] : f.c[i];
    return r;
}

// conversions (a value as another scalar type) and bitcasts (its bits)
template <typename R, typename T> __device__ __forceinline__ R wgsl_convert(T a) { return R(a); }
template <> __device__ __forceinline__ __half wgsl_convert<__half, float>(float a) { return __float2half_rn(a); }
template <> __device__ __forceinline__ float wgsl_convert<float, __half>(__half a) { return __half2float(a); }
template <> __device__ __forceinline__ uint wgsl_convert<uint, __half>(__half a) { return uint(__half2float(a)); }
template <> __device__ __forceinline__ int wgsl_convert<int, __half>(__half a) { return int(__half2float(a)); }
template <> __device__ __forceinline__ __half wgsl_convert<__half, uint>(uint a) { return __float2half_rn(float(a)); }
template <> __device__ __forceinline__ __half wgsl_convert<__half, int>(int a) { return __float2half_rn(float(a)); }
// (WGSL clamps a float converted to an integer to the integer's range; NaN gives 0: CUDA's casts round toward zero and
// saturate the same way, cvt.rzi.sat)
template <typename R, typename T, int N> __device__ __forceinline__ vec<R, N> wgsl_convert(const vec<T, N>& a) { return vmap<R>(a, [](T x) { return wgsl_convert<R>(x); }); }
template <typename R, typename T> __device__ __forceinline__ R wgsl_bitcast(const T& a) {
    static_assert(sizeof(R) == sizeof(T), "a bitcast keeps the size");
    R r;
    memcpy(&r, &a, sizeof(R));
    return r;
}

// the built-ins: scalars, then vectors through vmap
__device__ __forceinline__ float wgsl_abs(float x) { return fabsf(x); }
__device__ __forceinline__ int wgsl_abs(int x) { return x == int(0x80000000) ? x : (x < 0 ? -x : x); }
__device__ __forceinline__ uint wgsl_abs(uint x) { return x; }
__device__ __forceinline__ __half wgsl_abs(__half x) { return __habs(x); }
__device__ __forceinline__ float wgsl_min(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ float wgsl_max(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ __half wgsl_min(__half a, __half b) { return __hmin(a, b); }
__device__ __forceinline__ __half wgsl_max(__half a, __half b) { return __hmax(a, b); }
__device__ __forceinline__ int wgsl_min(int a, int b) { return min(a, b); }
__device__ __forceinline__ int wgsl_max(int a, int b) { return max(a, b); }
__device__ __forceinline__ uint wgsl_min(uint a, uint b) { return min(a, b); }
__device__ __forceinline__ uint wgsl_max(uint a, uint b) { return max(a, b); }
template <typename T> __device__ __forceinline__ T wgsl_clamp(T x, T lo, T hi) { return wgsl_min(wgsl_max(x, lo), hi); }
__device__ __forceinline__ float wgsl_sign(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }
__device__ __forceinline__ int wgsl_sign(int x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }
__device__ __forceinline__ float wgsl_saturate(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }
__device__ __forceinline__ float wgsl_step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
__device__ __forceinline__ float wgsl_mix(float a, float b, float t) { return a * (1.0f - t) + b * t; }
__device__ __forceinline__ float wgsl_fract(float x) { return x - floorf(x); }
__device__ __forceinline__ float wgsl_smoothstep(float lo, float hi, float x) { float t = fminf(fmaxf((x - lo) / (hi - lo), 0.0f), 1.0f); return t * t * (3.0f - 2.0f * t); }
__device__ __forceinline__ float wgsl_ldexp(float x, int e) { return ldexpf(x, e); }

#define WGSL_UNARY_F32(NAME, CALL) __device__ __forceinline__ float NAME(float x) { return CALL(x); }
WGSL_UNARY_F32(wgsl_exp, expf)
WGSL_UNARY_F32(wgsl_exp2, exp2f)
WGSL_UNARY_F32(wgsl_log, logf)
WGSL_UNARY_F32(wgsl_log2, log2f)
WGSL_UNARY_F32(wgsl_sqrt, sqrtf)
WGSL_UNARY_F32(wgsl_inverse_sqrt, rsqrtf)
WGSL_UNARY_F32(wgsl_sin, sinf)
WGSL_UNARY_F32(wgsl_cos, cosf)
WGSL_UNARY_F32(wgsl_tan, tanf)
WGSL_UNARY_F32(wgsl_tanh, tanhf)
WGSL_UNARY_F32(wgsl_sinh, sinhf)
WGSL_UNARY_F32(wgsl_cosh, coshf)
WGSL_UNARY_F32(wgsl_asin, asinf)
WGSL_UNARY_F32(wgsl_acos, acosf)
WGSL_UNARY_F32(wgsl_atan, atanf)
WGSL_UNARY_F32(wgsl_floor, floorf)
WGSL_UNARY_F32(wgsl_ceil, ceilf)
WGSL_UNARY_F32(wgsl_trunc, truncf)
WGSL_UNARY_F32(wgsl_round, rintf)
#undef WGSL_UNARY_F32
__device__ __forceinline__ float wgsl_pow(float a, float b) { return powf(a, b); }
__device__ __forceinline__ float wgsl_atan2(float a, float b) { return atan2f(a, b); }
__device__ __forceinline__ float wgsl_fma(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __forceinline__ __half wgsl_exp(__half x) { return hexp(x); }
__device__ __forceinline__ __half wgsl_sqrt(__half x) { return hsqrt(x); }
__device__ __forceinline__ __half wgsl_fma(__half a, __half b, __half c) { return __hfma(a, b, c); }
__device__ __forceinline__ uint wgsl_count_one_bits(uint x) { return __popc(x); }
__device__ __forceinline__ int wgsl_count_one_bits(int x) { return __popc(uint(x)); }
__device__ __forceinline__ uint wgsl_reverse_bits(uint x) { return __brev(x); }
__device__ __forceinline__ uint wgsl_count_leading_zeros(uint x) { return __clz(x); }
__device__ __forceinline__ uint wgsl_count_trailing_zeros(uint x) { return x == 0u ? 32u : uint(__ffs(x) - 1); }
__device__ __forceinline__ uint wgsl_first_trailing_bit(uint x) { return x == 0u ? 0xffffffffu : uint(__ffs(x) - 1); }
__device__ __forceinline__ uint wgsl_first_leading_bit(uint x) { return x == 0u ? 0xffffffffu : uint(31 - __clz(x)); }
__device__ __forceinline__ int wgsl_first_leading_bit(int x) { uint u = x < 0 ? ~uint(x) : uint(x); return u == 0u ? -1 : int(31 - __clz(u)); }
__device__ __forceinline__ uint wgsl_extract_bits(uint e, uint offset, uint count) {
    uint o = min(offset, 32u), c = min(count, 32u - o);
    return c == 0u ? 0u : (e >> o) & (c == 32u ? 0xffffffffu : ((1u << c) - 1u));
}
__device__ __forceinline__ int wgsl_extract_bits(int e, uint offset, uint count) {
    uint o = min(offset, 32u), c = min(count, 32u - o);
    if (c == 0u) return 0;
    return int(uint(e) << (32u - o - c)) >> (32u - c);
}
__device__ __forceinline__ uint wgsl_insert_bits(uint e, uint n, uint offset, uint count) {
    uint o = min(offset, 32u), c = min(count, 32u - o);
    if (c == 0u) return e;
    uint mask = (c == 32u ? 0xffffffffu : ((1u << c) - 1u)) << o;
    return (e & ~mask) | ((n << o) & mask);
}

// dot products: a vector's, and the packed int8 ones (dp4a)
template <typename T, int N> __device__ __forceinline__ T wgsl_dot(const vec<T, N>& a, const vec<T, N>& b) {
    T s = a.c[0] * b.c[0];
#pragma unroll
    for (int i = 1; i < N; i++) s = s + a.c[i] * b.c[i];
    return s;
}
__device__ __forceinline__ int wgsl_dot4_i8_packed(uint a, uint b) { return __dp4a(int(a), int(b), 0); }
__device__ __forceinline__ uint wgsl_dot4_u8_packed(uint a, uint b) { return __dp4a(a, b, 0u); }

// packing
__device__ __forceinline__ uint wgsl_pack2x16float(const vec<float, 2>& v) {
    return uint(__half_as_ushort(__float2half_rn(v.c[0]))) | (uint(__half_as_ushort(__float2half_rn(v.c[1]))) << 16);
}
__device__ __forceinline__ vec<float, 2> wgsl_unpack2x16float(uint w) {
    return vec<float, 2>(__half2float(__ushort_as_half((unsigned short)(w & 0xffffu))), __half2float(__ushort_as_half((unsigned short)(w >> 16))));
}
__device__ __forceinline__ uint wgsl_pack4x8unorm(const vec<float, 4>& v) {
    uint r = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) r |= uint(rintf(fminf(fmaxf(v.c[i], 0.0f), 1.0f) * 255.0f)) << (8 * i);
    return r;
}
__device__ __forceinline__ vec<float, 4> wgsl_unpack4x8unorm(uint w) {
    vec<float, 4> r;
#pragma unroll
    for (int i = 0; i < 4; i++) r.c[i] = float((w >> (8 * i)) & 255u) / 255.0f;
    return r;
}

// the relational built-ins
template <int N> __device__ __forceinline__ bool wgsl_all(const vec<bool, N>& v) {
    bool r = true;
#pragma unroll
    for (int i = 0; i < N; i++) r = r && v.c[i];
    return r;
}
template <int N> __device__ __forceinline__ bool wgsl_any(const vec<bool, N>& v) {
    bool r = false;
#pragma unroll
    for (int i = 0; i < N; i++) r = r || v.c[i];
    return r;
}
__device__ __forceinline__ bool wgsl_all(bool v) { return v; }
__device__ __forceinline__ bool wgsl_any(bool v) { return v; }
__device__ __forceinline__ bool wgsl_is_nan(float x) { return isnan(x); }
__device__ __forceinline__ bool wgsl_is_inf(float x) { return isinf(x); }

// the element-wise built-ins of vectors, from the scalars'
#define WGSL_VEC1(NAME)                                                                                       \
    template <typename T, int N> __device__ __forceinline__ auto NAME(const vec<T, N>& a) {                  \
        return vmap<decltype(NAME(a.c[0]))>(a, [](T x) { return NAME(x); });                                 \
    }
#define WGSL_VEC2(NAME)                                                                                                  \
    template <typename T, int N> __device__ __forceinline__ vec<T, N> NAME(const vec<T, N>& a, const vec<T, N>& b) {    \
        return vmap2<T>(a, b, [](T x, T y) { return NAME(x, y); });                                                   \
    }
#define WGSL_VEC3(NAME)                                                                                                                      \
    template <typename T, int N> __device__ __forceinline__ vec<T, N> NAME(const vec<T, N>& a, const vec<T, N>& b, const vec<T, N>& c) {    \
        vec<T, N> r;                                                                                                                         \
        for (int i = 0; i < N; i++) r.c[i] = NAME(a.c[i], b.c[i], c.c[i]);                                                                  \
        return r;                                                                                                                            \
    }
WGSL_VEC1(wgsl_abs) WGSL_VEC1(wgsl_sign) WGSL_VEC1(wgsl_saturate) WGSL_VEC1(wgsl_fract)
WGSL_VEC1(wgsl_exp) WGSL_VEC1(wgsl_exp2) WGSL_VEC1(wgsl_log) WGSL_VEC1(wgsl_log2) WGSL_VEC1(wgsl_sqrt) WGSL_VEC1(wgsl_inverse_sqrt)
WGSL_VEC1(wgsl_sin) WGSL_VEC1(wgsl_cos) WGSL_VEC1(wgsl_tan) WGSL_VEC1(wgsl_tanh) WGSL_VEC1(wgsl_sinh) WGSL_VEC1(wgsl_cosh)
WGSL_VEC1(wgsl_asin) WGSL_VEC1(wgsl_acos) WGSL_VEC1(wgsl_atan)
WGSL_VEC1(wgsl_floor) WGSL_VEC1(wgsl_ceil) WGSL_VEC1(wgsl_trunc) WGSL_VEC1(wgsl_round)
WGSL_VEC1(wgsl_count_one_bits) WGSL_VEC1(wgsl_reverse_bits) WGSL_VEC1(wgsl_first_leading_bit) WGSL_VEC1(wgsl_first_trailing_bit)
WGSL_VEC1(wgsl_is_nan) WGSL_VEC1(wgsl_is_inf)
WGSL_VEC2(wgsl_min) WGSL_VEC2(wgsl_max) WGSL_VEC2(wgsl_pow) WGSL_VEC2(wgsl_atan2) WGSL_VEC2(wgsl_step)
WGSL_VEC3(wgsl_clamp) WGSL_VEC3(wgsl_fma) WGSL_VEC3(wgsl_mix) WGSL_VEC3(wgsl_smoothstep)
#undef WGSL_VEC1
#undef WGSL_VEC2
#undef WGSL_VEC3
template <typename T, int N> __device__ __forceinline__ vec<T, N> wgsl_mix(const vec<T, N>& a, const vec<T, N>& b, T t) { return vmap2<T>(a, b, [t](T x, T y) { return wgsl_mix(x, y, t); }); }

// an index kept in its array (WebGPU's robust access, as naga's Restrict policy: the last element for one past it,
// a negative one included), and a runtime-sized array's length from its binding's size in bytes
__device__ __forceinline__ uint wgsl_index(uint i, uint len) { return min(i, len == 0u ? 0u : len - 1u); }
__device__ __forceinline__ uint wgsl_index(int i, uint len) { return wgsl_index(uint(i), len); }
__device__ __forceinline__ uint wgsl_length(uint bytes, uint start, uint stride) { return bytes > start ? (bytes - start) / stride : 0u; }
// an array's elements from index i on (none past its end)
__device__ __forceinline__ uint wgsl_avail(uint i, uint len) { return i < len ? len - i : 0u; }
__device__ __forceinline__ uint wgsl_avail(int i, uint len) { return wgsl_avail(uint(i), len); }

// the invocation's own index in its workgroup, as WGSL counts it. The workgroup's size is the kernel's constants
// (WGSL_WG_X, _Y, _Z, defined before this prelude): a runtime need not fill CUDA's blockDim, which tinygrad's launches
// leave as 0 (its own kernels never read it)
__device__ __forceinline__ uint wgsl_local_index() { return threadIdx.x + WGSL_WG_X * (threadIdx.y + WGSL_WG_Y * threadIdx.z); }

// a workgroup variable zeroed before the kernel runs (WebGPU's default): every invocation a part of it
__device__ __forceinline__ void wgsl_zero_workgroup(void* p, uint bytes) {
    uint* w = (uint*)p;
    uint words = bytes / 4u, n = WGSL_WG_X * WGSL_WG_Y * WGSL_WG_Z;
    for (uint i = wgsl_local_index(); i < words; i += n) w[i] = 0u;
    for (uint i = words * 4u + wgsl_local_index(); i < bytes; i += n) ((unsigned char*)p)[i] = 0;
}
