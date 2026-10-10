// mma.h: CUDA's wmma (the tensor cores' fragments) on the CPU, for cuda_emu.h's kernels. Each invocation holds a
// fragment's whole 16 x 16 matrix and does its warp's op itself: a warp's invocations all do alike, as WGSL's
// cooperative ops are in uniform control flow, so every one has the warp's result. The sums are f32's, rounded to an
// f16 accumulator's as each op ends.
#pragma once
#include <type_traits>

namespace nvcuda {
namespace wmma {

struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };

// rows by columns: an A's M x K, a B's K x N, an accumulator's M x N
template <typename Use, int M, int N, int K, typename T, typename Layout = void> struct fragment {
    typedef T element_type;
    enum { num_elements = 256 };
    T x[16][16];
};

template <typename Use, typename T, typename Layout>
inline void load_matrix_sync(fragment<Use, 16, 16, 16, T, Layout>& f, const T* p, unsigned ldm) {
    for (unsigned r = 0; r < 16; r++)
        for (unsigned c = 0; c < 16; c++) f.x[r][c] = std::is_same<Layout, row_major>::value ? p[r * ldm + c] : p[c * ldm + r];
}

template <typename T> inline void load_matrix_sync(fragment<accumulator, 16, 16, 16, T>& f, const T* p, unsigned ldm, layout_t layout) {
    for (unsigned r = 0; r < 16; r++)
        for (unsigned c = 0; c < 16; c++) f.x[r][c] = layout == mem_row_major ? p[r * ldm + c] : p[c * ldm + r];
}

template <typename T> inline void store_matrix_sync(T* p, const fragment<accumulator, 16, 16, 16, T>& f, unsigned ldm, layout_t layout) {
    for (unsigned r = 0; r < 16; r++)
        for (unsigned c = 0; c < 16; c++) (layout == mem_row_major ? p[r * ldm + c] : p[c * ldm + r]) = f.x[r][c];
}

template <typename Use, typename T, typename Layout> inline void fill_fragment(fragment<Use, 16, 16, 16, T, Layout>& f, const T& v) {
    for (unsigned r = 0; r < 16; r++)
        for (unsigned c = 0; c < 16; c++) f.x[r][c] = v;
}

template <typename TA, typename LA, typename TB, typename LB, typename TC>
inline void mma_sync(fragment<accumulator, 16, 16, 16, TC>& d, const fragment<matrix_a, 16, 16, 16, TA, LA>& a, const fragment<matrix_b, 16, 16, 16, TB, LB>& b,
                     const fragment<accumulator, 16, 16, 16, TC>& c) {
    float sum[16][16];
    for (unsigned i = 0; i < 16; i++)
        for (unsigned j = 0; j < 16; j++) {
            float s = float(c.x[i][j]);
            for (unsigned k = 0; k < 16; k++) s += float(a.x[i][k]) * float(b.x[k][j]);
            sum[i][j] = s;
        }
    for (unsigned i = 0; i < 16; i++)
        for (unsigned j = 0; j < 16; j++) d.x[i][j] = TC(sum[i][j]);
}

}  // namespace wmma
}  // namespace nvcuda
