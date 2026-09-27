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
EXL3_GEMV(80)
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
EXL3_TILE(80)
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

// VENDORED-LOCAL (Qwen3.8-Flash-Next): grouped MoE over EXL3 experts. A layer's (token,
// expert) assignments are sorted by expert into segments; each launch covers every
// selected expert. Rounding follows exl3_had and the tile GEMV exactly.
__device__ __forceinline__ void exl3_butterfly(float* v, int t) {
    for(int s=1;s<128;s*=2) {
        float p=v[t], q=v[t^s];
        __syncthreads();
        v[t]=(t&s) ? q-p : p+q;
        __syncthreads();
    }
}
// Input transform of the gate (z=0) and up (z=1) projections for each assignment.
extern "C" __global__ void exl3_moe_in(const float* x, const unsigned int* rows, const unsigned int* experts,
    const float* suh0, const float* suh1, float* out0, float* out1, int K) {
    __shared__ float v[128];
    int t=threadIdx.x, col=blockIdx.x*128+t, a=blockIdx.y;
    const float* suh=blockIdx.z ? suh1 : suh0;
    float* out=blockIdx.z ? out1 : out0;
    unsigned int e=experts[a];
    v[t]=exl3_half(x[(size_t)rows[a]*K+col])*suh[(size_t)e*K+col];
    __syncthreads();
    exl3_butterfly(v,t);
    out[(size_t)a*K+col]=exl3_half(v[t]*0.08838834764831845f);
}
// Tile GEMV over each segment's rows, eight at a time: every weight tile is decoded once per
// eight rows. seg = [start | count | expert], `stride` apart; `*nseg` segments are live.
template<int TW, int R, int W> __device__ void exl3_moe_tile(const unsigned int* we, const float* x, float* y,
    int start, int count, int K, int N) {
    const int NW=TW/2, B=TW/16;
    int lane=threadIdx.x, warp=threadIdx.y, nt=blockIdx.x;
    int first=(lane*8+1)*B + (TW%16==8 ? (lane*8+1)/2 : 0);
    int st=first-16;
    if(st<0) st+=NW*32;
    int wi=st/32, offset=st%32;
    int next=wi+1<NW?wi+1:wi+1-NW;
    int last=wi+2<NW?wi+2:wi+2-NW;
    // This lane's four input rows of a 16x16 tile: b, b+1, b+8, b+9.
    int base=(lane%4)*2;
    __shared__ float sums[W][R][16];
    for(int r0=0;r0<count;r0+=R) {
        int rows=min(R,count-r0);
        float lo[R], hi[R];
        #pragma unroll
        for(int m=0;m<R;++m) { lo[m]=0.f; hi[m]=0.f; }
        for(int kt=warp;kt<K/16;kt+=W) {
            const unsigned int* p=we+((size_t)kt*(N/16)+nt)*NW;
            unsigned int a=p[wi],b=p[next],c=p[last];
            unsigned int high=__funnelshift_lc(b,a,offset);
            unsigned int low=__funnelshift_lc(c,b,offset);
            float wv[8];
            #pragma unroll
            for(int j=0;j<8;++j) {
                const int delta=j*B + (TW%16==8 ? (j+1)/2 : 0);
                unsigned int code=(delta<32 ? __funnelshift_lc(low,high,delta) : low<<(delta-32))>>16;
                unsigned int vv=code*0x83dcd12dU;
                unsigned int sum=exl3_byte_sum(vv);
                wv[j]=exl3_half(fmaf(1024.0f+float(sum),0.00676727294921875f,-10.3828125f));
            }
            // j -> input row base + (j%2) + 8*((j/2)%2); j<4 feeds lo, j>=4 hi.
            #pragma unroll
            for(int m=0;m<R;++m) {
                if(m<rows) {
                    const float* xr=x+(size_t)(start+r0+m)*K+kt*16+base;
                    float2 u=*(const float2*)xr, v=*(const float2*)(xr+8);
                    lo[m]=fmaf(u.x,wv[0],lo[m]); lo[m]=fmaf(u.y,wv[1],lo[m]); lo[m]=fmaf(v.x,wv[2],lo[m]); lo[m]=fmaf(v.y,wv[3],lo[m]);
                    hi[m]=fmaf(u.x,wv[4],hi[m]); hi[m]=fmaf(u.y,wv[5],hi[m]); hi[m]=fmaf(v.x,wv[6],hi[m]); hi[m]=fmaf(v.y,wv[7],hi[m]);
                }
            }
        }
        #pragma unroll
        for(int m=0;m<R;++m) {
            lo[m]+=__shfl_xor_sync(0xffffffff,lo[m],1); hi[m]+=__shfl_xor_sync(0xffffffff,hi[m],1);
            lo[m]+=__shfl_xor_sync(0xffffffff,lo[m],2); hi[m]+=__shfl_xor_sync(0xffffffff,hi[m],2);
            if(lane%4==0) { sums[warp][m][lane/4]=lo[m]; sums[warp][m][lane/4+8]=hi[m]; }
        }
        __syncthreads();
        if(warp==0 && lane<16) {
            for(int m=0;m<rows;++m) {
                float t=0.f;
                #pragma unroll
                for(int w=0;w<W;++w) t+=sums[w][m][lane];
                y[(size_t)(start+r0+m)*N+nt*16+lane]=t;
            }
        }
        __syncthreads();
    }
}
// z picks the matrix (gate or up; the down launch has one). Each expert has its own words
// offset and rate (the routed experts 3-bit, the shared one 5-bit).
extern "C" __global__ void exl3_moe_tile(const unsigned int* w0, const unsigned int* w1,
    const unsigned long long* off0, const unsigned long long* off1, const int* tw0, const int* tw1,
    const float* x0, const float* x1, float* y0, float* y1,
    const unsigned int* seg, const unsigned int* nseg, int stride, int K, int N, int one_row) {
    int sg=blockIdx.y;
    if(sg>=(int)*nseg) return;
    int one=blockIdx.z;
    unsigned int e=seg[2*stride+sg];
    const unsigned int* we=(one?w1:w0)+(one?off1:off0)[e];
    int tw=(one?tw1:tw0)[e];
    int start=seg[sg], count=seg[stride+sg];
    // One token (decode): every segment has one row, and the one-row variant skips the rest.
    if(one_row) {
        if(tw==48) exl3_moe_tile<48,1,8>(we, one?x1:x0, one?y1:y0, start, count, K, N);
        else exl3_moe_tile<80,1,8>(we, one?x1:x0, one?y1:y0, start, count, K, N);
    } else {
        if(tw==48) exl3_moe_tile<48,16,8>(we, one?x1:x0, one?y1:y0, start, count, K, N);
        else exl3_moe_tile<80,16,8>(we, one?x1:x0, one?y1:y0, start, count, K, N);
    }
}
// Each token's experts: its top_k by logit (their softmax the weights), then the shared
// expert, weighted by sigmoid of the last logit. A thread per logit.
__device__ void exl3_moe_route_row(const float* logits, unsigned int* ae, float* aw, int routed, int top_k, int r) {
    int t=threadIdx.x;
    const float* l=logits+(size_t)r*(routed+1);
    __shared__ float sv[1024];
    __shared__ int pick[32];
    __shared__ float pv[32];
    // Each logit's rank: how many beat it (ties to the lower index). The top_k go by rank.
    if(t<routed) sv[t]=l[t];
    __syncthreads();
    if(t<routed) {
        float v=sv[t];
        int rank=0;
        for(int i=0;i<routed;++i) { float o=sv[i]; rank+=(o>v)||(o==v&&i<t); }
        if(rank<top_k) { pick[rank]=t; pv[rank]=v; }
    }
    __syncthreads();
    if(t==0) {
        int per=top_k+1;
        float sum=0.f;
        for(int j=0;j<top_k;++j) sum+=expf(pv[j]-pv[0]);
        for(int j=0;j<top_k;++j) { ae[(size_t)r*per+j]=pick[j]; aw[(size_t)r*per+j]=expf(pv[j]-pv[0])/sum; }
        ae[(size_t)r*per+top_k]=routed;
        aw[(size_t)r*per+top_k]=1.f/(1.f+expf(-l[routed]));
    }
    __syncthreads();
}
// One block per token.
extern "C" __global__ void exl3_moe_route(const float* logits, unsigned int* ae, float* aw, int routed, int top_k) {
    exl3_moe_route_row(logits, ae, aw, routed, top_k, blockIdx.x);
}
// Sort the assignments by expert (a counting sort in one block): the sorted rows, experts and
// weights, and the segments: each expert's rows in chunks of at most `chunk`, so a busy
// expert (the shared one takes every token) spreads over many blocks. seg = [start | count |
// expert], `stride` apart.
__device__ void exl3_moe_group_block(const unsigned int* ae, const float* aw, int total, int per_row, int experts,
    int chunk, int stride, unsigned int* rows, unsigned int* ex, float* w, unsigned int* seg, unsigned int* nseg, unsigned int* slot);
extern "C" __global__ void exl3_moe_group(const unsigned int* ae, const float* aw, int total, int per_row, int experts,
    int chunk, int stride, unsigned int* rows, unsigned int* ex, float* w, unsigned int* seg, unsigned int* nseg, unsigned int* slot) {
    exl3_moe_group_block(ae, aw, total, per_row, experts, chunk, stride, rows, ex, w, seg, nseg, slot);
}
// A few tokens (decode): routing then grouping in one block, one launch. Their assignments
// (at most blockDim) sort by expert, then by assignment, each counting what goes before it.
extern "C" __global__ void exl3_moe_route_group(const float* logits, unsigned int* ae, float* aw, int routed, int top_k, int token_rows,
    int chunk, int stride, unsigned int* rows, unsigned int* ex, float* w, unsigned int* seg, unsigned int* nseg, unsigned int* slot) {
    for(int r=0;r<token_rows;++r) exl3_moe_route_row(logits, ae, aw, routed, top_k, r);
    __shared__ unsigned int se[1024];
    __shared__ int first[1024], many[1024], starts[1024];
    int t=threadIdx.x, per=top_k+1, total=token_rows*per;
    if(t<total) se[t]=ae[t];
    __syncthreads();
    int pos=0;
    if(t<total) {
        unsigned int e=se[t];
        int before=0, same=0, earlier=0;
        for(int b=0;b<total;++b) { unsigned int o=se[b]; before+=o<e; same+=o==e; earlier+=(o==e)&&(b<t); }
        pos=before+earlier;
        rows[pos]=t/per; ex[pos]=e; w[pos]=aw[t]; slot[t]=pos;
        first[pos]=before; many[pos]=same;
    }
    __syncthreads();
    // Each expert's rows in chunks of at most `chunk`: a segment starts every chunk rows.
    int start=t<total && (t-first[t])%chunk==0;
    if(t<total) starts[t]=start;
    __syncthreads();
    if(start) {
        int s=0;
        for(int q=0;q<t;++q) s+=starts[q];
        seg[s]=t; seg[stride+s]=min(chunk, many[t]-(t-first[t])); seg[2*stride+s]=ex[t];
    }
    if(t==0) { int n=0; for(int q=0;q<total;++q) n+=starts[q]; *nseg=n; }
}
__device__ void exl3_moe_group_block(const unsigned int* ae, const float* aw, int total, int per_row, int experts,
    int chunk, int stride, unsigned int* rows, unsigned int* ex, float* w, unsigned int* seg, unsigned int* nseg, unsigned int* slot) {
    extern __shared__ int group_bins[];
    int* cnt=group_bins;
    int* cur=group_bins+experts;
    int t=threadIdx.x;
    for(int i=t;i<experts;i+=blockDim.x) cnt[i]=0;
    __syncthreads();
    for(int a=t;a<total;a+=blockDim.x) atomicAdd(&cnt[ae[a]],1);
    __syncthreads();
    // Exclusive scans (Hillis-Steele, a thread an expert: experts <= blockDim) of the row counts
    // and of the chunk counts give each expert its first row and first segment.
    __shared__ int rows_before[1024], segs_before[1024];
    int mine=t<experts ? cnt[t] : 0;
    int chunks=(mine+chunk-1)/chunk;
    rows_before[t]=mine; segs_before[t]=chunks;
    __syncthreads();
    for(int o=1;o<blockDim.x;o<<=1) {
        int a=t>=o ? rows_before[t-o] : 0, b=t>=o ? segs_before[t-o] : 0;
        __syncthreads();
        rows_before[t]+=a; segs_before[t]+=b;
        __syncthreads();
    }
    if(t<experts) {
        int first=rows_before[t]-mine, s=segs_before[t]-chunks;
        cur[t]=first;
        for(int o=0;o<mine;o+=chunk,++s) { seg[s]=first+o; seg[stride+s]=min(chunk,mine-o); seg[2*stride+s]=t; }
    }
    if(t==blockDim.x-1) *nseg=segs_before[t];
    __syncthreads();
    for(int a=t;a<total;a+=blockDim.x) {
        unsigned int e=ae[a];
        int at=atomicAdd(&cur[e],1);
        rows[at]=a/per_row; ex[at]=e; w[at]=aw[a];
        slot[a]=at;
    }
}
// Gate and up output transforms, silu(gate) * up, then the down projection's input transform.
// dg, du (optional): LoRA deltas for gate and up; hk (optional): the activation kept for the
// down projection's LoRA.
extern "C" __global__ void exl3_moe_mid(const float* yg, const float* yu, const float* svh_g, const float* svh_u,
    const float* suh_d, const unsigned int* experts, float* xd, int N, const float* dg, const float* du, float* hk) {
    __shared__ float v[128], u[128];
    int t=threadIdx.x, col=blockIdx.x*128+t, a=blockIdx.y;
    unsigned int e=experts[a];
    v[t]=exl3_half(yg[(size_t)a*N+col]);
    u[t]=exl3_half(yu[(size_t)a*N+col]);
    __syncthreads();
    exl3_butterfly(v,t);
    exl3_butterfly(u,t);
    float g=exl3_half(v[t]*0.08838834764831845f*svh_g[(size_t)e*N+col]);
    float up=exl3_half(u[t]*0.08838834764831845f*svh_u[(size_t)e*N+col]);
    if(dg) g+=dg[(size_t)a*N+col];
    if(du) up+=du[(size_t)a*N+col];
    float h=g/(1.0f+expf(-g))*up;
    if(hk) hk[(size_t)a*N+col]=h;
    __syncthreads();
    v[t]=exl3_half(h)*suh_d[(size_t)e*N+col];
    __syncthreads();
    exl3_butterfly(v,t);
    xd[(size_t)a*N+col]=exl3_half(v[t]*0.08838834764831845f);
}
// The down projection's output transform, weighted, in place: a block per 128 columns of an
// assignment's row.
extern "C" __global__ void exl3_moe_out(float* yd, const float* svh, const unsigned int* experts, const float* w, int N) {
    __shared__ float v[128];
    int t=threadIdx.x, col=blockIdx.x*128+t, a=blockIdx.y;
    unsigned int e=experts[a];
    v[t]=exl3_half(yd[(size_t)a*N+col]);
    __syncthreads();
    exl3_butterfly(v,t);
    yd[(size_t)a*N+col]=w[a]*exl3_half(v[t]*0.08838834764831845f*svh[(size_t)e*N+col]);
}
// VENDORED-LOCAL: LoRA on the experts: for assignment a (expert e, `slot_of[e]` its LoRA or
// none), out[a] = B_s (A_s x) with x the input row (rows[a] of x, or a), weighted by w[a] and
// added when accumulating, else written (zero without a LoRA). A [slots, R, K], B [slots, N, R].
extern "C" __global__ void exl3_moe_lora(const float* x, const unsigned int* rows, const unsigned int* experts, const unsigned int* slot_of,
    const float* A, const float* B, const float* w, float* out, int K, int N, int R, int accumulate) {
    extern __shared__ float lora_t[];
    int a=blockIdx.x, tid=threadIdx.x, lane=tid&31, warp=tid>>5, warps=blockDim.x>>5;
    unsigned int s=slot_of[experts[a]];
    float* o=out+(size_t)a*N;
    if(s==0xffffffffu) {
        if(!accumulate) for(int c=tid;c<N;c+=blockDim.x) o[c]=0.f;
        return;
    }
    const float* xr=x+(size_t)(rows ? rows[a] : a)*K;
    const float* As=A+(size_t)s*R*K;
    for(int r=warp;r<R;r+=warps) {
        float acc=0.f;
        for(int i=lane;i<K;i+=32) acc+=As[(size_t)r*K+i]*xr[i];
        for(int off=16;off>0;off>>=1) acc+=__shfl_xor_sync(0xffffffff,acc,off);
        if(lane==0) lora_t[r]=acc;
    }
    __syncthreads();
    const float* Bs=B+(size_t)s*N*R;
    float wa=w ? w[a] : 1.f;
    for(int c=tid;c<N;c+=blockDim.x) {
        float acc=0.f;
        for(int r=0;r<R;++r) acc+=Bs[(size_t)c*R+r]*lora_t[r];
        if(accumulate) o[c]+=wa*acc; else o[c]=acc;
    }
}
// Each token's experts summed in the token's own order (`slot` finds each one's sorted row):
// the same sum every run.
extern "C" __global__ void exl3_moe_sum(const float* yd, const unsigned int* slot, float* out, int rows, int N, int per_row) {
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long)rows*N) return;
    long r=i/N; int col=(int)(i%N);
    float acc=0.f;
    for(int j=0;j<per_row;++j) acc+=yd[(size_t)slot[r*per_row+j]*N+col];
    out[i]=acc;
}
