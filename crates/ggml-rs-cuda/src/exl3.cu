// VENDORED-LOCAL: EXL3 mul1, from the documented exllamav3 tile/trellis layout.
// Original packed weights stay resident; prefill reconstructs one projection
// at a time as temporary scratch. Half-rates alternate k and k+1 bits.
__device__ float exl3_half(float x) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    float rounded;
    asm("cvt.f32.f16 %0, %1;" : "=f"(rounded) : "h"(h));
    return rounded;
}
// Exact integer byte sum; DP4A replaces seven scalar bit/add operations.
__device__ __forceinline__ unsigned int exl3_byte_sum(unsigned int x) {
#if __CUDA_ARCH__ >= 610
    unsigned int sum;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(sum) : "r"(x), "r"(0x01010101U), "r"(0U));
    return sum;
#else
    return (x&255)+((x>>8)&255)+((x>>16)&255)+(x>>24);
#endif
}
__device__ __forceinline__ float exl3_value_rate(const unsigned int* w, int k, int n, int N, int tw) {
    int r=k&15, c=n&15;
    int i=((r%8/2)+4*(c%8))*8+(r%2)+2*(r/8)+4*(c/8);
    int end=(i+1)*(tw/16) + ((tw%16)==8 ? (i+1)/2 : 0);
    int nw=tw/2, start=(end+nw*32-16)%(nw*32);
    const unsigned int* p=w+((size_t)(k/16)*(N/16)+n/16)*nw;
    unsigned long long v=((unsigned long long)p[start/32]<<32)|p[(start/32+1)%nw];
    unsigned int code=(v>>(48-start%32))&65535;
    unsigned int x=code*0x83dcd12dU;
    unsigned int sum=exl3_byte_sum(x);
    return exl3_half(fmaf(1024.0f + float(sum), 0.00676727294921875f, -10.3828125f));
}
template<int TW> __device__ __forceinline__ float exl3_value_fixed(const unsigned int* w, int k, int n, int N) {
    return exl3_value_rate(w,k,n,N,TW);
}
__device__ float exl3_value(const unsigned int* w, int k, int n, int N, int tw) {
    if(tw==32) return exl3_value_fixed<32>(w,k,n,N);
    if(tw==48) return exl3_value_fixed<48>(w,k,n,N);
    if(tw==56) return exl3_value_fixed<56>(w,k,n,N);
    if(tw==64) return exl3_value_fixed<64>(w,k,n,N);
    if(tw==96) return exl3_value_fixed<96>(w,k,n,N);
    return exl3_value_rate(w,k,n,N,tw);
}
extern "C" __global__ void exl3_had(const float* x, float* y, const float* scale,
    const unsigned int* map, int width, int post) {
    __shared__ float v[128];
    int t=threadIdx.x, col=blockIdx.x*128+t, row=blockIdx.y;
    v[t]=post ? exl3_half(x[(size_t)row*width+col]) : exl3_half(x[(size_t)row*width+map[col]])*scale[col];
    __syncthreads();
    for(int s=1;s<128;s*=2) {
        float a=v[t], b=v[t^s];
        __syncthreads();
        v[t]=(t&s) ? b-a : a+b;
        __syncthreads();
    }
    float val=exl3_half(v[t]*0.08838834764831845f*(post?scale[col]:1.0f));
    // Output map here is original -> caller (the inverse of output_map).
    y[(size_t)row*width+(post?map[col]:col)]=val;
}
extern "C" __global__ void exl3_reconstruct(const unsigned int* w,float* out,int K,int N,int tw) {
    size_t p=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(p<(size_t)K*N) out[p]=exl3_value(w,p%K,p/K,N,tw);
}
template<int TW> __device__ void exl3_gemv_fixed(const unsigned int* w,const float* x,float* y,int K,int N) {
    int n=blockIdx.x*8+threadIdx.y, lane=threadIdx.x;
    float sum=0;
    if(n<N) for(int k=lane;k<K;k+=32) sum=fmaf(x[k],exl3_value_fixed<TW>(w,k,n,N),sum);
    for(int d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(n<N && lane==0)y[n]=sum;
}
#define EXL3_GEMV(TW) \
extern "C" __global__ void exl3_gemv_##TW(const unsigned int* w,const float* x,float* y,int K,int N) { \
    exl3_gemv_fixed<TW>(w,x,y,K,N); \
}
EXL3_GEMV(32)
EXL3_GEMV(48)
EXL3_GEMV(56)
EXL3_GEMV(64)
EXL3_GEMV(96)

// Decode an entire 16x16 packed tile cooperatively. One lane owns its eight
// consecutive trellis steps; three coalesced word loads replace sixteen
// scattered loads. Four lanes reduce each output; four warps partition K.
template<int TW> __device__ void exl3_gemv_tile(const unsigned int* w,const float* x,float* y,int K,int N) {
    const int NW=TW/2, B=TW/16;
    int lane=threadIdx.x, warp=threadIdx.y, nt=blockIdx.x;
    int split=blockIdx.y, splits=gridDim.y;
    float lo=0,hi=0;
    int first=(lane*8+1)*B + (TW%16==8 ? (lane*8+1)/2 : 0);
    int start=first-16;
    if(start<0) start+=NW*32;
    int wi=start/32, offset=start%32;
    int next=wi+1<NW?wi+1:wi+1-NW;
    int last=wi+2<NW?wi+2:wi+2-NW;
    for(int kt=warp+split*4;kt<K/16;kt+=4*splits) {
        const unsigned int* p=w+((size_t)kt*(N/16)+nt)*NW;
        unsigned int a=p[wi],b=p[next],c=p[last];
        // Align once for all eight overlapping 16-bit windows. Subsequent
        // offsets are compile-time constants, including half-bit rates.
        unsigned int high=__funnelshift_lc(b,a,offset);
        unsigned int low=__funnelshift_lc(c,b,offset);
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int delta=j*B + (TW%16==8 ? (j+1)/2 : 0);
            unsigned int code=(delta<32 ? __funnelshift_lc(low,high,delta) : low<<(delta-32))>>16;
            unsigned int v=code*0x83dcd12dU;
            unsigned int sum=exl3_byte_sum(v);
            float weight=exl3_half(fmaf(1024.0f+float(sum),0.00676727294921875f,-10.3828125f));
            int r=(lane%4)*2+(j%2)+8*((j/2)%2);
            float input=x[kt*16+r];
            if(j<4) lo=fmaf(input,weight,lo); else hi=fmaf(input,weight,hi);
        }
    }
    lo+=__shfl_xor_sync(0xffffffff,lo,1); hi+=__shfl_xor_sync(0xffffffff,hi,1);
    lo+=__shfl_xor_sync(0xffffffff,lo,2); hi+=__shfl_xor_sync(0xffffffff,hi,2);
    __shared__ float sums[4][16];
    if(lane%4==0) { sums[warp][lane/4]=lo; sums[warp][lane/4+8]=hi; }
    __syncthreads();
    if(warp==0 && lane<16) y[(size_t)split*N+nt*16+lane]=sums[0][lane]+sums[1][lane]+sums[2][lane]+sums[3][lane];
}
#define EXL3_TILE(TW) \
extern "C" __global__ void exl3_tile_##TW(const unsigned int* w,const float* x,float* y,int K,int N) { \
    exl3_gemv_tile<TW>(w,x,y,K,N); \
}
EXL3_TILE(32)
EXL3_TILE(48)
EXL3_TILE(56)
EXL3_TILE(64)
EXL3_TILE(96)

// Other valid EXL3 rates retain the scalar layout fallback.
extern "C" __global__ void exl3_gemv_generic(const unsigned int* w,const float* x,float* y,int K,int N,int tw) {
    int n=blockIdx.x*8+threadIdx.y, lane=threadIdx.x;
    float sum=0;
    if(n<N) for(int k=lane;k<K;k+=32) sum=fmaf(x[k],exl3_value_rate(w,k,n,N,tw),sum);
    for(int d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(n<N && lane==0)y[n]=sum;
}

// Reduce split-K partials before the original rounded output Hadamard.
// This fuses the reduction into an existing launch instead of adding one.
extern "C" __global__ void exl3_had_reduce(const float* x, float* y, const float* scale,
    const unsigned int* map, int width, int splits) {
    __shared__ float v[128];
    int t=threadIdx.x, col=blockIdx.x*128+t;
    float sum=0;
    for(int part=0;part<splits;++part) sum+=x[(size_t)part*width+col];
    v[t]=exl3_half(sum);
    __syncthreads();
    for(int step=1;step<128;step*=2) {
        float a=v[t],b=v[t^step];
        __syncthreads();
        v[t]=(t&step)?b-a:a+b;
        __syncthreads();
    }
    y[map[col]]=exl3_half(v[t]*0.08838834764831845f*scale[col]);
}
