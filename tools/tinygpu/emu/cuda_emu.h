// cuda_emu.h: a kernel crates/wgsl-cuda translated, compiled for the CPU and run there (webgpu_server.py --emulate), so
// a translation is checked without the card: a kernel that faults costs the card its link (docs/TINYGPU.md).
//
// Included before the kernel (clang++ -include), with this directory first on the include path (its cuda_fp16.h is
// empty). CUDA's words are the CPU's: __half is _Float16, atomics are the compiler's, threadIdx and blockIdx a thread's
// own. A workgroup runs on one thread, each invocation a fiber of its own stack: __syncthreads() goes to the next
// invocation, and the workgroup goes on once every one has reached it. Workgroup memory (__shared__) is the thread's
// (thread_local), so workgroups run on several threads at once, each its own.
#pragma once
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <type_traits>
#include <utility>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__ __restrict
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)
#define __shared__ static thread_local

struct uint3 {
    unsigned x, y, z;
};

inline thread_local uint3 emu_thread_idx, emu_block_idx, emu_grid_dim;
#define threadIdx emu_thread_idx
#define blockIdx emu_block_idx
#define gridDim emu_grid_dim
#define blockDim (uint3{WGSL_WG_X, WGSL_WG_Y, WGSL_WG_Z})

// ---- numbers ----

typedef _Float16 __half;
inline float __half2float(__half h) { return float(h); }
inline __half __float2half(float f) { return __half(f); }
inline __half __float2half_rn(float f) { return __half(f); }
inline unsigned short __half_as_ushort(__half h) { unsigned short u; memcpy(&u, &h, 2); return u; }
inline __half __ushort_as_half(unsigned short u) { __half h; memcpy(&h, &u, 2); return h; }
inline __half __habs(__half h) { return __half(fabsf(float(h))); }
inline __half __hmin(__half a, __half b) { return __half(fminf(float(a), float(b))); }
inline __half __hmax(__half a, __half b) { return __half(fmaxf(float(a), float(b))); }
inline __half __hfma(__half a, __half b, __half c) { return __half(fmaf(float(a), float(b), float(c))); }
inline __half hexp(__half h) { return __half(expf(float(h))); }
inline __half hsqrt(__half h) { return __half(sqrtf(float(h))); }
inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }

inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }

inline int __popc(unsigned x) { return __builtin_popcount(x); }
inline unsigned __brev(unsigned x) { return __builtin_bitreverse32(x); }
inline int __clz(unsigned x) { return x == 0 ? 32 : __builtin_clz(x); }
inline int __clz(int x) { return __clz(unsigned(x)); }
inline int __ffs(unsigned x) { return __builtin_ffs(int(x)); }
inline int __ffs(int x) { return __builtin_ffs(x); }

inline int __dp4a(int a, int b, int c) {
    for (int i = 0; i < 4; i++) c += int(int8_t(a >> (8 * i))) * int(int8_t(b >> (8 * i)));
    return c;
}
inline unsigned __dp4a(unsigned a, unsigned b, unsigned c) {
    for (int i = 0; i < 4; i++) c += ((a >> (8 * i)) & 255u) * ((b >> (8 * i)) & 255u);
    return c;
}

// ---- atomics (workgroups run on several threads at once; a workgroup's own invocations never interleave) ----

template <typename T> inline T atomicAdd(T* p, T v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicSub(T* p, T v) { return __atomic_fetch_sub(p, v, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicAnd(T* p, T v) { return __atomic_fetch_and(p, v, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicOr(T* p, T v) { return __atomic_fetch_or(p, v, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicXor(T* p, T v) { return __atomic_fetch_xor(p, v, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicMin(T* p, T v) { return __atomic_fetch_min(p, v, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicMax(T* p, T v) { return __atomic_fetch_max(p, v, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicExch(T* p, T v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicCAS(T* p, T expected, T v) {
    __atomic_compare_exchange_n(p, &expected, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expected;
}
inline void __threadfence() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
inline void __threadfence_block() { __atomic_signal_fence(__ATOMIC_SEQ_CST); }

// ---- fibers ----

extern "C" void emu_switch(void** save, void* to);  // save this stack's pointer, go to another's
extern "C" void emu_fiber_start();                  // a new fiber's first frame: the function of the argument

#if defined(__APPLE__)
#define EMU_SYM(name) "_" #name
#define EMU_LOCAL(name) ".private_extern _" #name "\n"
#else
#define EMU_SYM(name) #name
#define EMU_LOCAL(name) ".hidden " #name "\n"
#endif
#if defined(__aarch64__)
// the callee-saved registers (x19-x30, d8-d15) in a 160-byte frame on the stack left; a new fiber's x19 is its
// argument and x20 its function
#define EMU_FRAME 160
#define EMU_ARG 0
#define EMU_FN 1
#define EMU_RET 11
asm(".text\n.p2align 2\n" EMU_LOCAL(emu_switch) EMU_SYM(emu_switch) ":\n"
    "  sub sp, sp, #160\n"
    "  stp x19, x20, [sp, #0]\n  stp x21, x22, [sp, #16]\n  stp x23, x24, [sp, #32]\n"
    "  stp x25, x26, [sp, #48]\n  stp x27, x28, [sp, #64]\n  stp x29, x30, [sp, #80]\n"
    "  stp d8, d9, [sp, #96]\n  stp d10, d11, [sp, #112]\n  stp d12, d13, [sp, #128]\n  stp d14, d15, [sp, #144]\n"
    "  mov x9, sp\n  str x9, [x0]\n  mov sp, x1\n"
    "  ldp x19, x20, [sp, #0]\n  ldp x21, x22, [sp, #16]\n  ldp x23, x24, [sp, #32]\n"
    "  ldp x25, x26, [sp, #48]\n  ldp x27, x28, [sp, #64]\n  ldp x29, x30, [sp, #80]\n"
    "  ldp d8, d9, [sp, #96]\n  ldp d10, d11, [sp, #112]\n  ldp d12, d13, [sp, #128]\n  ldp d14, d15, [sp, #144]\n"
    "  add sp, sp, #160\n  ret\n"
    ".p2align 2\n" EMU_LOCAL(emu_fiber_start) EMU_SYM(emu_fiber_start) ":\n"
    "  mov x0, x19\n  blr x20\n  brk #0\n");
#elif defined(__x86_64__)
// the System V callee-saved registers (r15, r14, r13, r12, rbx, rbp) and the return address; a new fiber's r12 is its
// argument and r13 its function
#define EMU_FRAME 56
#define EMU_ARG 3
#define EMU_FN 2
#define EMU_RET 6
asm(".text\n.p2align 4\n" EMU_LOCAL(emu_switch) EMU_SYM(emu_switch) ":\n"
    "  pushq %rbp\n  pushq %rbx\n  pushq %r12\n  pushq %r13\n  pushq %r14\n  pushq %r15\n"
    "  movq %rsp, (%rdi)\n  movq %rsi, %rsp\n"
    "  popq %r15\n  popq %r14\n  popq %r13\n  popq %r12\n  popq %rbx\n  popq %rbp\n  ret\n"
    ".p2align 4\n" EMU_LOCAL(emu_fiber_start) EMU_SYM(emu_fiber_start) ":\n"
    "  movq %r12, %rdi\n  callq *%r13\n  ud2\n");
#else
#error "cuda_emu.h: fibers for arm64 and x86-64"
#endif

namespace emu {

typedef void (*Call)(void** args);

struct Workgroup {
    Call call;
    void** args;
    unsigned n, cur;
    void** sp;    // each invocation's stack, where it stopped
    bool* done;
    void* sched;  // the scheduler's
};
inline thread_local Workgroup* wg;

inline void fiber_main(void*) {
    Workgroup* w = wg;
    w->call(w->args);
    w = wg;
    w->done[w->cur] = true;
    emu_switch(&w->sp[w->cur], w->sched);
    __builtin_trap();
}

inline void* new_fiber(char* top) {
    void** f = (void**)(top - EMU_FRAME);
    memset(f, 0, EMU_FRAME);
    f[EMU_ARG] = nullptr;
    f[EMU_FN] = (void*)&fiber_main;
    f[EMU_RET] = (void*)&emu_fiber_start;   // where emu_switch returns to
    return f;
}

template <typename... P, size_t... I> inline void call_with(void (*f)(P...), void** a, std::index_sequence<I...>) { f((P)a[I]...); }
template <typename... P> inline void call(void (*f)(P...), void** a) { call_with(f, a, std::index_sequence_for<P...>{}); }

// the workgroups first .. first + count of the grid, in turn on this thread; `stacks` is room for a stack of
// `stack_size` bytes an invocation (its lowest page may be a guard)
inline void run(Call call, void** args, uint3 grid, uint3 size, uint64_t first, uint64_t count, bool barriers, char* stacks, size_t stack_size) {
    const unsigned n = size.x * size.y * size.z;
    emu_grid_dim = grid;
    void* sp[1024];
    bool done[1024];
    Workgroup w{call, args, n, 0, sp, done, nullptr};
    for (uint64_t b = first; b < first + count; b++) {
        emu_block_idx = uint3{unsigned(b % grid.x), unsigned((b / grid.x) % grid.y), unsigned(b / (uint64_t(grid.x) * grid.y))};
        if (!barriers) {
            for (unsigned i = 0; i < n; i++) {
                emu_thread_idx = uint3{i % size.x, (i / size.x) % size.y, i / (size.x * size.y)};
                call(args);
            }
            continue;
        }
        for (unsigned i = 0; i < n; i++) {
            sp[i] = new_fiber(stacks + size_t(i + 1) * stack_size);
            done[i] = false;
        }
        wg = &w;
        for (unsigned alive = n; alive;) {
            for (unsigned i = 0; i < n; i++) {
                if (done[i]) continue;
                w.cur = i;
                emu_thread_idx = uint3{i % size.x, (i / size.x) % size.y, i / (size.x * size.y)};
                emu_switch(&w.sched, sp[i]);
                if (done[i]) alive--;
            }
        }
        wg = nullptr;
    }
}

}  // namespace emu

inline void __syncthreads() {
    emu::Workgroup* w = emu::wg;
    emu_switch(&w->sp[w->cur], w->sched);
}

// the kernel's launcher, after it: emu_launch(args, grid, first, count, stacks, stack_size)
#define EMU_LAUNCHER(KERNEL, BARRIERS)                                                                                  \
    extern "C" void emu_launch(void** args, unsigned gx, unsigned gy, unsigned gz, uint64_t first, uint64_t count,      \
                               char* stacks, size_t stack_size) {                                                       \
        emu::run(+[](void** a) { emu::call(KERNEL, a); }, args, uint3{gx, gy, gz}, uint3{WGSL_WG_X, WGSL_WG_Y, WGSL_WG_Z}, \
                 first, count, BARRIERS, stacks, stack_size);                                                           \
    }
