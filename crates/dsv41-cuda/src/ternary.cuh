// Custom ternary kernels; no claim of native BitNet integer activation arithmetic.
// We retain OAIY's FP8-simulated activations and BF16 rounding boundaries.
__device__ __forceinline__ float ternary_part(const float* x, const unsigned char* w,
                                              const unsigned char* s, int k, int lane, int lanes) {
    float acc=0.f;
    for (int b=lane; b<k/32; b+=lanes) {
        unsigned long long q=*(const unsigned long long*)(w+b*8);
        float part=0.f;
        float xf[32]; load32(x+b*32, aligned16(x), xf);
#pragma unroll
        for (int p=0;p<16;p++) {
            int a=(int)((q>>(p*4))&3)-1;
            int z=(int)((q>>(p*4+2))&3)-1;
            part += xf[2*p]*(float)a + xf[2*p+1]*(float)z;
        }
        unsigned short bits=(unsigned short)s[(b/4)*2] | ((unsigned short)s[(b/4)*2+1]<<8);
        float scale;
        asm("cvt.f32.f16 %0, %1;" : "=f"(scale) : "h"(bits));
        acc += part*scale;
    }
    return acc;
}

__global__ void gemv_ternary(const float* x,const unsigned char* w,const unsigned char* s,float* y,
                            int n,int k,int nt,int round) {
    int row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32, lane=threadIdx.x&31;
    if(row>=n)return;
    for(int t=0;t<nt;t++) {
        float v=ternary_part(x+(long)t*k,w+(long)row*k/4,s+(long)row*k/64,k,lane,32);
        v=warp_sum(v);
        if(lane==0)y[(long)t*n+row]=round?to_bf16(v):v;
    }
}

__global__ void moe_gate_up_ternary(const float* x,const unsigned long long* recs,float* h,
                                    int nexp,int inter,int dim,float lim) {
    int item=blockIdx.x*(blockDim.x/32)+threadIdx.x/32,lane=threadIdx.x&31;
    if(item>=nexp*inter)return;
    int e=item/inter,r=item%inter;
    const unsigned char* rec=(const unsigned char*)recs[e];
    long W=(long)inter*dim/4,S=(long)inter*dim/64;
    float a=ternary_part(x,rec+(long)r*dim/4,rec+3*W+(long)r*dim/64,dim,lane,32);
    float z=ternary_part(x,rec+2*W+(long)r*dim/4,rec+3*W+2*S+(long)r*dim/64,dim,lane,32);
    a=warp_sum(a);z=warp_sum(z);
    if(lane==0) {
        float g=to_bf16(a),u=to_bf16(z);
        if(lim>0.f){u=fminf(fmaxf(u,-lim),lim);g=fminf(g,lim);}
        float rw=__uint_as_float((unsigned int)recs[2*nexp+e]);
        h[item]=to_bf16(g/(1.f+expf(-g))*u*rw);
    }
}

__global__ void moe_down_ternary(const float* x,const unsigned long long* recs,float* out,int nexp,int inter,int dim) {
    int item=(blockIdx.x*blockDim.x+threadIdx.x)/16, lane=threadIdx.x&15;
    bool on=item<nexp*dim;
    float a=0.f; int e=on?item/dim:0,r=on?item%dim:0;
    if(on) {
        const unsigned char* rec=(const unsigned char*)recs[e];
        long W=(long)inter*dim/4,S=(long)inter*dim/64;
        a=ternary_part(x+(long)e*inter,rec+W+(long)r*inter/4,rec+3*W+S+(long)r*inter/64,inter,lane,16);
    }
    for(int o=8;o>0;o>>=1)a+=__shfl_xor_sync(0xffffffffu,a,o);
    if(on && lane==0)out[(long)recs[nexp+e]*dim+r]=to_bf16(a);
}
