// VENDORED-LOCAL: Split-K attention: each block normalizes a bounded 4096-token partition.
// A second kernel merges its (maximum, denominator, weighted value) exactly
// with log-sum-exp scaling. No repeated GQA K/V or context-sized score matrix.
extern "C" __global__ void attention_partition_f32(
    const float* q, const float* k, const float* v, float* partial,
    int seq, int nh, int nkv, int len, int hd, float scale, int past, int window) {
    const int chunk = 4096;
    int h = blockIdx.x, s = blockIdx.y, part = blockIdx.z;
    int parts = (len + chunk - 1) / chunk;
    int begin = part * chunk, end = min(begin + chunk, min(len, past + s + 1));
    if (window > 0) begin = max(begin, past + s + 1 - window);
    int count = max(0, end - begin), tid = threadIdx.x, bs = blockDim.x;
    int kh = h / (nh / nkv);
    extern __shared__ float shared[];
    float* scores = shared;
    float* reduce = shared + chunk;
    const float* qr = q + (s * nh + h) * hd;
    float* dst = partial + ((s * nh + h) * parts + part) * (hd + 2);
    for (int t = tid; t < count; t += bs) {
        const float* kr = k + ((begin + t) * nkv + kh) * hd;
        float dot = 0;
        for (int d = 0; d < hd; ++d) dot += qr[d] * kr[d];
        scores[t] = dot * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int t = tid; t < count; t += bs) mx = fmaxf(mx, scores[t]);
    reduce[tid] = mx;
    __syncthreads();
    for (int r = bs / 2; r; r >>= 1) {
        if (tid < r) reduce[tid] = fmaxf(reduce[tid], reduce[tid + r]);
        __syncthreads();
    }
    mx = reduce[0];
    __syncthreads();
    float sum = 0;
    for (int t = tid; t < count; t += bs) {
        float p = expf(scores[t] - mx);
        scores[t] = p;
        sum += p;
    }
    reduce[tid] = sum;
    __syncthreads();
    for (int r = bs / 2; r; r >>= 1) {
        if (tid < r) reduce[tid] += reduce[tid + r];
        __syncthreads();
    }
    if (tid == 0) { dst[hd] = mx; dst[hd + 1] = reduce[0]; }
    for (int d = tid; d < hd; d += bs) {
        float acc = 0;
        for (int t = 0; t < count; ++t)
            acc += scores[t] * v[((begin + t) * nkv + kh) * hd + d];
        dst[d] = acc;
    }
}

extern "C" __global__ void attention_merge_f32(
    const float* partial, float* out, int rows, int parts, int hd) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float* src = partial + row * parts * (hd + 2);
    float mx = -INFINITY;
    for (int p = 0; p < parts; ++p) mx = fmaxf(mx, src[p * (hd + 2) + hd]);
    float sum = 0;
    for (int p = 0; p < parts; ++p) {
        const float* item = src + p * (hd + 2);
        if (item[hd + 1] > 0) sum += expf(item[hd] - mx) * item[hd + 1];
    }
    for (int d = tid; d < hd; d += blockDim.x) {
        float val = 0;
        for (int p = 0; p < parts; ++p) {
            const float* item = src + p * (hd + 2);
            if (item[hd + 1] > 0) val += expf(item[hd] - mx) * item[d];
        }
        out[row * hd + d] = sum > 0 ? val / sum : 0;
    }
}
