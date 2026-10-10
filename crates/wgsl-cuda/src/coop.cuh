// wgsl-cuda's cooperative matrices: WGSL's coop_mat16x16 and its ops as CUDA's wmma fragments, on the tensor cores.
// After prelude.cuh, in a kernel that has any (crate wgsl-cuda's COOP_PRELUDE).
//
// An A's or B's layout is its fragment's type (row_major: coopLoadT's row-major memory; col_major: coopLoad's), a C's
// is the load's and store's. Every op is the subgroup's (a warp's) together, as WGSL's are: in its uniform control flow.
//
// A fragment is 16 rows (or columns) of 16 from its pointer, `stride` apart; `avail` is what its array has from there.
// Where the fragment fits, wmma reads and writes it in place; where it would reach past the array (the last rows of a
// matrix whose buffer is not padded to them), it is staged through 256 elements of the warp's own (its slot of the
// kernel's scratch on the card, a local array on the CPU): an element past the array is read as 0 and not written, as
// WebGPU's robust access has it. In place, such a fragment would read or write past the buffer, and the card's memory
// past a buffer may be no one's: a write there faults the card.
#include <mma.h>
namespace wmma = nvcuda::wmma;

// in place where the fragment is all in its array and as wmma takes it: 32-byte aligned, its stride a multiple of
// 16 bytes (SPIR-V's and Metal's loads take any; wmma's address another's memory past those)
template <typename T> __device__ __forceinline__ bool wgsl_coop_fits(const T* p, uint stride, uint avail) {
    return 15u * stride + 16u <= avail && ((unsigned long long)p & 31u) == 0u && (stride * sizeof(T)) % 16u == 0u;
}

#ifdef __CUDA_ARCH__
// the warp's own 1 KB of the scratch: 64 warps' slots a unit (SM), as the unit numbers its warps
__device__ __forceinline__ void* wgsl_coop_slot(float* scratch) {
    uint sm, warp;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp));
    return scratch + (sm * 64u + warp) * 256u;
}
#define WGSL_COOP_STAGE(T, s, scratch) T* s = (T*)wgsl_coop_slot(scratch)
#define WGSL_COOP_EACH(e) for (uint e = wgsl_local_index() & 31u; e < 256u; e += 32u)
#define WGSL_COOP_SYNC() __syncwarp()
#else
#include <stdio.h>
// (on the CPU each invocation does its warp's whole op itself: its own staging, every element, no warp to wait for)
#define WGSL_COOP_STAGE(T, s, scratch) T s[256]; wgsl_coop_staged()
#define WGSL_COOP_EACH(e) for (uint e = 0u; e < 256u; e++)
#define WGSL_COOP_SYNC()
inline void wgsl_coop_staged() {
    static bool said = false;
    const char* t = getenv("TINYGPU_EMU_TRACE");
    if (!said && t && t[0] == '1') fprintf(stderr, "emu: a cooperative fragment past its array or not as wmma takes it (aligned): staged\n");
    said = true;
}
#endif

template <typename Use, typename Layout, typename T>
__device__ __forceinline__ wmma::fragment<Use, 16, 16, 16, T, Layout> wgsl_coop_load(const T* p, uint stride, uint avail, float* scratch) {
    wmma::fragment<Use, 16, 16, 16, T, Layout> f;
    if (wgsl_coop_fits(p, stride, avail)) {
        wmma::load_matrix_sync(f, p, stride);
        return f;
    }
    WGSL_COOP_STAGE(T, s, scratch);
    WGSL_COOP_EACH(e) {
        uint at = (e >> 4) * stride + (e & 15u);
        s[e] = at < avail ? p[at] : T(0.0f);
    }
    WGSL_COOP_SYNC();
    wmma::load_matrix_sync(f, s, 16u);
    WGSL_COOP_SYNC();
    return f;
}

template <typename T>
__device__ __forceinline__ wmma::fragment<wmma::accumulator, 16, 16, 16, T> wgsl_coop_load_c(const T* p, uint stride, bool row_major, uint avail, float* scratch) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, T> f;
    const wmma::layout_t layout = row_major ? wmma::mem_row_major : wmma::mem_col_major;
    if (wgsl_coop_fits(p, stride, avail)) {
        wmma::load_matrix_sync(f, p, stride, layout);
        return f;
    }
    WGSL_COOP_STAGE(T, s, scratch);
    WGSL_COOP_EACH(e) {
        uint at = (e >> 4) * stride + (e & 15u);
        s[e] = at < avail ? p[at] : T(0.0f);
    }
    WGSL_COOP_SYNC();
    wmma::load_matrix_sync(f, s, 16u, layout);
    WGSL_COOP_SYNC();
    return f;
}

template <typename A, typename B, typename C>
__device__ __forceinline__ C wgsl_coop_mma(const A& a, const B& b, const C& c) {
    C d;
    wmma::mma_sync(d, a, b, c);
    return d;
}

template <typename F> __device__ __forceinline__ F wgsl_coop_zero() {
    F f;
    wmma::fill_fragment(f, static_cast<typename F::element_type>(0.0f));
    return f;
}

template <typename T, typename F> __device__ __forceinline__ void wgsl_coop_store(T* p, const F& f, uint stride, bool row_major, uint avail, float* scratch) {
    const wmma::layout_t layout = row_major ? wmma::mem_row_major : wmma::mem_col_major;
    if (wgsl_coop_fits(p, stride, avail)) {
        wmma::store_matrix_sync(p, f, stride, layout);
        return;
    }
    WGSL_COOP_STAGE(T, s, scratch);
    WGSL_COOP_SYNC();
    wmma::store_matrix_sync(s, f, 16u, layout);
    WGSL_COOP_SYNC();
    WGSL_COOP_EACH(e) {
        uint at = (e >> 4) * stride + (e & 15u);
        if (at < avail) p[at] = s[e];
    }
    WGSL_COOP_SYNC();
}
