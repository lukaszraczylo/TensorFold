// Inkling 2 Mini's DFlash2 drafter (incoai/Inkling-2-Mini-DFlash2, Apache-2.0): per-stream context rings, the block's two-tap
// convs, attention and SwiGLU, the head's top 16 a row and the candidate selector's greedy path. After z-lab/dflash (MIT),
// this fork's GLM kernels (glm_dflash.cu) and MiaAI-Lab's patch 0018 (Apache-2.0), at Mini's widths.
#include <cuda_bf16.h>
#include <stdint.h>

namespace {
constexpr int D = 2304;
constexpr int HD = 128;
constexpr int KVH = 8;
constexpr int QH = 32;
constexpr int KVW = KVH * HD;
constexpr int WIN = 2048;
constexpr int BLOCK = 8;
constexpr int GROUP = 16;
constexpr int GROUPS = D / GROUP;
constexpr int CH = 16;
constexpr int RANK = 256;
constexpr int TOP = 16;
constexpr int CHUNK_COLS = 4096;

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 tobf(float x) { return __float2bfloat16_rn(x); }

// The ring of one layer: K then V, [WIN][KVW] each, layers one after another.
__device__ __forceinline__ __nv_bfloat16* ringK(uint64_t base, int layer) {
    return reinterpret_cast<__nv_bfloat16*>(base) + (long long)layer * 2 * WIN * KVW;
}
__device__ __forceinline__ __nv_bfloat16* ringV(uint64_t base, int layer) { return ringK(base, layer) + (long long)WIN * KVW; }

// Qwen3's per-head RMSNorm then rotate-half RoPE of one head (64 threads: dims d and d + 64), in place or to out.
__device__ __forceinline__ void normRope(const __nv_bfloat16* x, const __nv_bfloat16* w, float eps, long long pos, float theta,
                                         __nv_bfloat16* out, float* red) {
    const int d = threadIdx.x;  // 0..63
    const float a = bf(x[d]), b = bf(x[d + 64]);
    float s = a * a + b * b;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((d & 31) == 0) red[d >> 5] = s;
    __syncthreads();
    const float rinv = rsqrtf((red[0] + red[1]) / HD + eps);
    __syncthreads();
    const float na = bf(tobf(bf(w[d]) * bf(tobf(a * rinv))));
    const float nb = bf(tobf(bf(w[d + 64]) * bf(tobf(b * rinv))));
    const float inv = powf(theta, -(float)(2 * d) / HD);
    const float ang = (float)pos * inv;
    const float c = cosf(ang), sn = sinf(ang);
    out[d] = tobf(na * c - nb * sn);
    out[d + 64] = tobf(nb * c + na * sn);
}
}  // namespace


// out = w * bf16(x * rsqrt(mean(x^2) + eps)) per row of D; 256 threads a row.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_norm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                                __nv_bfloat16* __restrict__ out, float eps) {
    __shared__ float red[8];
    const long long r = blockIdx.x;
    float s = 0.f;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        const float v = bf(x[r * D + c]);
        s += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < (int)(blockDim.x >> 5); ++k) t += red[k];
    const float rinv = rsqrtf(t / D + eps);
    for (int c = threadIdx.x; c < D; c += blockDim.x) out[r * D + c] = tobf(bf(w[c]) * bf(tobf(bf(x[r * D + c]) * rinv)));
}

// The block's queries (fused qkv, stride ldq) normed and rotated at pos[r] into q, keys into kb, values into vb.
extern "C" __global__ void tf_ink2d_block_qkv(const __nv_bfloat16* __restrict__ qkv, int ldq, const __nv_bfloat16* __restrict__ qn,
                                        const __nv_bfloat16* __restrict__ kn, const int* __restrict__ pos, float eps, float theta,
                                        __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ kb, __nv_bfloat16* __restrict__ vb) {
        __shared__ float red[2];
    const int r = blockIdx.x, h = blockIdx.y;
    const __nv_bfloat16* row = qkv + (long long)r * ldq;
    if (h < QH) {
        normRope(row + h * HD, qn, eps, pos[r], theta, q + (long long)r * QH * HD + h * HD, red);
    } else {
        const int kh = h - QH;
        normRope(row + QH * HD + kh * HD, kn, eps, pos[r], theta, kb + (long long)r * KVW + kh * HD, red);
        const __nv_bfloat16* v = row + QH * HD + KVW + kh * HD;
        __nv_bfloat16* vo = vb + (long long)r * KVW + kh * HD;
        vo[threadIdx.x] = v[threadIdx.x];
        vo[threadIdx.x + 64] = v[threadIdx.x + 64];
    }
}

// Context rows' keys and values (k | v) normed, rotated at pos0 + r, into ``ring`` at slot position % WIN of ``layer``.
extern "C" __global__ void tf_ink2d_ctx_kv(const __nv_bfloat16* __restrict__ kv, const __nv_bfloat16* __restrict__ kn,
                                           long long pos0, uint64_t ring, int layer, float eps, float theta) {
    __shared__ float red[2];
    const int r = blockIdx.x, kh = blockIdx.y;
    const long long p = pos0 + r;
    const int slot = (int)(p % WIN);
    const __nv_bfloat16* row = kv + (long long)r * 2 * KVW;
    normRope(row + kh * HD, kn, eps, p, theta, ringK(ring, layer) + (long long)slot * KVW + kh * HD, red);
    __nv_bfloat16* vo = ringV(ring, layer) + (long long)slot * KVW + kh * HD;
    vo[threadIdx.x] = row[KVW + kh * HD + threadIdx.x];
    vo[threadIdx.x + 64] = row[KVW + kh * HD + threadIdx.x + 64];
}

// Two-tap dynamic causal conv over a block: y[t] = sum_o (base[o] + dyn[t][o][c / 16]) * z[t - o]; add or store.
extern "C" __global__ void tf_ink2d_conv(const __nv_bfloat16* __restrict__ z, const __nv_bfloat16* __restrict__ dyn, int which,
                                   const __nv_bfloat16* __restrict__ base, __nv_bfloat16* __restrict__ out, int add) {
        const int r = blockIdx.x, t = r % BLOCK;
    const __nv_bfloat16* dr = dyn + (long long)r * 4 * GROUPS + which * 2 * GROUPS;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024 && c < D; c += blockDim.x) {
        const int g = c / GROUP;
        float y = (bf(base[c]) + bf(dr[g])) * bf(z[(long long)r * D + c]);
        if (t > 0) y += (bf(base[D + c]) + bf(dr[GROUPS + g])) * bf(z[(long long)(r - 1) * D + c]);
        __nv_bfloat16* o = out + (long long)r * D + c;
        *o = add ? tobf(bf(*o) + bf(tobf(y))) : tobf(y);
    }
}

// SwiGLU over [gate | up] rows (width F each): bf16(bf16(silu(g)) * u). grid (R, ceil(F / 1024)), 256 threads.
extern "C" __global__ void tf_ink2d_swiglu(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ out, int F) {
        const int r = blockIdx.x;
    const __nv_bfloat16* g = gu + (long long)r * 2 * F;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024 && c < F; c += blockDim.x) {
        const float x = bf(g[c]);
        const float s = bf(tobf(x / (1.0f + expf(-x))));
        out[(long long)r * F + c] = tobf(s * bf(g[F + c]));
    }
}

// tf_ink2d_attn split over CH key chunks (running max, sum, fp32 output), merged by tf_ink2d_attn_merge: not latency-bound.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_attn_part(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
        const __nv_bfloat16* __restrict__ vb, const uint64_t* __restrict__ rings, const long long* __restrict__ starts,
        const long long* __restrict__ froms, int layer, float scale, float* __restrict__ pm, float* __restrict__ pl,
        float* __restrict__ pacc) {
        const int s = blockIdx.x, kh = blockIdx.y, c = blockIdx.z;
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = s * BLOCK + j;
    const long long start = starts[s], p = start + j;
    long long lo = p - (WIN - 1);
    if (lo < froms[s]) lo = froms[s];
    if (lo < 0) lo = 0;
    const long long n_ctx = start > lo ? start - lo : 0;
    const long long T = n_ctx + BLOCK, L = (T + CH - 1) / CH;
    const long long n0 = (long long)c * L, n1 = n0 + L < T ? n0 + L : T;
    float qv[4][4], m[4], l[4], acc[4][4];
    for (int h = 0; h < 4; ++h) {
        const __nv_bfloat16* qh = q + (long long)r * QH * HD + (kh * 4 + h) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) qv[h][i] = bf(qh[i]) * scale;
        m[h] = -INFINITY;
        l[h] = 0.f;
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
    }
    const __nv_bfloat16* K = ringK(rings[s], layer);
    const __nv_bfloat16* V = ringV(rings[s], layer);
    for (long long n = n0; n < n1; ++n) {
        const __nv_bfloat16 *kr, *vr;
        if (n < n_ctx) {
            const int slot = (int)((lo + n) % WIN);
            kr = K + (long long)slot * KVW + kh * HD + lane * 4;
            vr = V + (long long)slot * KVW + kh * HD + lane * 4;
        } else {
            const long long rb = (long long)s * BLOCK + (n - n_ctx);
            kr = kb + rb * KVW + kh * HD + lane * 4;
            vr = vb + rb * KVW + kh * HD + lane * 4;
        }
        float kf[4], vf[4];
        for (int i = 0; i < 4; ++i) {
            kf[i] = bf(kr[i]);
            vf[i] = bf(vr[i]);
        }
        for (int h = 0; h < 4; ++h) {
            float d = qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float mn = fmaxf(m[h], d);
            const float corr = __expf(m[h] - mn), pe = __expf(d - mn);
            l[h] = l[h] * corr + pe;
            for (int i = 0; i < 4; ++i) acc[h][i] = acc[h][i] * corr + pe * vf[i];
            m[h] = mn;
        }
    }
    for (int h = 0; h < 4; ++h) {
        const long long at = ((long long)r * QH + kh * 4 + h) * CH + c;
        float* pa = pacc + at * HD + lane * 4;
        for (int i = 0; i < 4; ++i) pa[i] = acc[h][i];
        if (lane == 0) {
            pm[at] = m[h];   // -inf with l = 0 for a chunk past the row's keys: the merge weighs it 0
            pl[at] = l[h];
        }
    }
}

// The chunks' partials of row r, head h into o[r][h] (bf16). grid (R, QH), 128 threads (a thread a dim).
extern "C" __global__ void tf_ink2d_attn_merge(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                         __nv_bfloat16* __restrict__ o) {
        const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const long long base = ((long long)r * QH + h) * CH;
    float m = -INFINITY;
    for (int c = 0; c < CH; ++c) m = fmaxf(m, pm[base + c]);
    float l = 0.f, a = 0.f;
    for (int c = 0; c < CH; ++c) {
        const float w = __expf(pm[base + c] - m);
        l += pl[base + c] * w;
        a += pacc[(base + c) * HD + d] * w;
    }
    o[(long long)r * QH * HD + h * HD + d] = tobf(a / l);
}

namespace {
__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

// The block's best (value, index) of sv/si [n] (256 threads), the winner's slot cleared to -inf.
__device__ void takeBest(float* sv, int* si, int n, float* rv, int* ri, float* outv, int* outi) {
    float bv = -INFINITY;
    int bi = 0x7fffffff, bs = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        if (better(sv[j], si[j], bv, bi)) { bv = sv[j]; bi = si[j]; bs = j; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o), os = __shfl_xor_sync(0xffffffffu, bs, o);
        if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; bs = os; }
    }
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) { rv[w] = bv; ri[w] = bi; ri[8 + w] = bs; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int best = 0;
        for (int k = 1; k < (int)(blockDim.x >> 5); ++k)
            if (better(rv[k], ri[k], rv[best], ri[best])) best = k;
        *outv = rv[best];
        *outi = ri[best];
        if (ri[8 + best] >= 0) sv[ri[8 + best]] = -INFINITY;
    }
    __syncthreads();
}
}  // namespace

// A row's 16 largest logits (fp32 [rows][cols]) per 4096-column chunk as ids and values; ties keep the low id.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_top16_part(const float* __restrict__ logits, int cols,
                                                                      float* __restrict__ pv, int* __restrict__ pi) {
    __shared__ float sv[CHUNK_COLS];
    __shared__ int si[CHUNK_COLS];
    __shared__ float rv[8];
    __shared__ int ri[16];
    const int r = blockIdx.x, c0 = blockIdx.y * CHUNK_COLS;
    const int n = min(CHUNK_COLS, cols - c0);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = logits[(long long)r * cols + c0 + j];
        si[j] = c0 + j;
    }
    __syncthreads();
    const long long base = ((long long)r * gridDim.y + blockIdx.y) * TOP;
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, pv + base + t, pi + base + t);
}

// The chunks' 16 best merged into a row's: out[r][0..16) ids, out[r][16..32) the values' bits. chunks <= 64.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_top16_merge(const float* __restrict__ pv, const int* __restrict__ pi, int chunks,
                                                                       int* __restrict__ out) {
    __shared__ float sv[64 * TOP];
    __shared__ int si[64 * TOP];
    __shared__ float rv[8];
    __shared__ int ri[16];
    __shared__ float ov[TOP];
    __shared__ int oi[TOP];
    const int r = blockIdx.x, n = chunks * TOP;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = pv[(long long)r * n + j];
        si[j] = pi[(long long)r * n + j];
    }
    __syncthreads();
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, ov + t, oi + t);
    if (threadIdx.x < TOP) {
        out[r * 2 * TOP + threadIdx.x] = oi[threadIdx.x];
        out[r * 2 * TOP + TOP + threadIdx.x] = __float_as_int(ov[threadIdx.x]);
    }
}

// The candidate selector's greedy path for stream s: row t takes the candidate maximizing logit + <pred[prev] * g[t], succ[c]>
// (prev: the anchor ids[s * 8], then the path so far). walk[0..8) is the anchor then the drafts; walk[8..16) holds each draft's
// own probability: the softmax over the 16 candidates of cap * tanh(score / cap) (the trained "after_correction" cap).
// cand [S * 7][2 * TOP] (rows 1..7 of a block) as tf_ink2d_top16_merge writes, g fp32 [S * 8][RANK], pred and succ bf16 [V][RANK]. grid (S), 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_select(const int* __restrict__ ids, const int* __restrict__ cand,
                                                                  const float* __restrict__ g, const __nv_bfloat16* __restrict__ pred,
                                                                  const __nv_bfloat16* __restrict__ succ,
                                                                  const uint64_t* __restrict__ walks, float cap) {
    __shared__ float pg[RANK];
    __shared__ float sc[TOP];
    __shared__ int prev_s;
    const int s = blockIdx.x, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    int* walk = reinterpret_cast<int*>(walks[s]);
    float* chances = reinterpret_cast<float*>(walk + BLOCK);
    if (tid == 0) {
        prev_s = ids[s * BLOCK];
        walk[0] = prev_s;
        chances[0] = 1.0f;
    }
    __syncthreads();
    for (int t = 1; t < BLOCK; ++t) {
        const int r = s * (BLOCK - 1) + t - 1;
        pg[tid] = bf(pred[(long long)prev_s * RANK + tid]) * g[(long long)(s * BLOCK + t) * RANK + tid];
        __syncthreads();
        for (int k = 0; k < 2; ++k) {
            const int c = w * 2 + k;
            const __nv_bfloat16* S = succ + (long long)cand[r * 2 * TOP + c] * RANK;
            float d = 0.f;
            for (int i = lane; i < RANK; i += 32) d += pg[i] * bf(S[i]);
            for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
            if (lane == 0) sc[c] = __int_as_float(cand[r * 2 * TOP + TOP + c]) + d;
        }
        __syncthreads();
        if (tid == 0) {
            int best = 0;
            for (int c = 1; c < TOP; ++c)
                if (sc[c] > sc[best]) best = c;
            const float top = cap * tanhf(sc[best] / cap);
            float z = 0.f;
            for (int c = 0; c < TOP; ++c) z += expf(cap * tanhf(sc[c] / cap) - top);
            const int id = cand[r * 2 * TOP + best];
            walk[t] = id;
            chances[t] = 1.0f / z;
            prev_s = id;
        }
        __syncthreads();
    }
}

// The target head's NVFP4 copy for draft candidates: out[r][n] = sum_k a[r][k] . w[n][k], fp32, rows <= BLOCK, one warp a
// weight row. Codes are e2m1 pairs [N][D/2] (low nibble the even element), scales e4m3 [N][D/16], one global scale gs.
// The e2m1 and e4m3 grids and the block-16 layout are NVFP4's; the row-major packing is ours.
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_head4(const __nv_bfloat16* __restrict__ a, int rows,
                                                                 const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                                                                 float gs, int N, float* __restrict__ out) {
    __shared__ __align__(16) __nv_bfloat16 sa[BLOCK * D];
    for (int i = threadIdx.x; i < rows * D; i += blockDim.x) sa[i] = a[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= N) return;
    const uint8_t* wc = codes + (long long)n * (D / 2);
    const uint8_t* ws = scales + (long long)n * (D / 16);
    const float grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float acc[BLOCK];
#pragma unroll
    for (int r = 0; r < BLOCK; ++r) acc[r] = 0.f;
#pragma unroll
    for (int i = 0; i < D / 256; ++i) {
        const int e0 = (i * 32 + lane) * 8;
        const uint32_t word = *reinterpret_cast<const uint32_t*>(wc + e0 / 2);
        const uint32_t sc = ws[e0 / 16];
        const int ex = (sc >> 3) & 15, mant = sc & 7;
        const float sv = (ex == 0 ? mant * (1.f / 8.f) * (1.f / 64.f) : (1.f + mant * (1.f / 8.f)) * exp2f((float)(ex - 7))) * gs;
        float w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t nib = (word >> (4 * j)) & 15u;
            const float v = grid[nib & 7];
            w[j] = (nib & 8u ? -v : v) * sv;
        }
#pragma unroll
        for (int r = 0; r < BLOCK; ++r) {
            if (r >= rows) break;
            const uint4 raw = *reinterpret_cast<const uint4*>(sa + r * D + e0);
            const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
            float s = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) s += w[j] * bf(h[j]);
            acc[r] += s;
        }
    }
#pragma unroll
    for (int r = 0; r < BLOCK; ++r) {
        float v = acc[r];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && r < rows) out[(long long)r * N + n] = v;
    }
}

// Quantising the target head to the NVFP4 layout tf_ink2d_head4 reads, on the device (ModelOpt's recipe: global scale
// amax / (6 * 448), an e4m3 scale a 16 inputs at the block's amax / 6, e2m1 codes to the nearest value).
__device__ __forceinline__ float e4m3Value(int c) {
    const int ex = c >> 3, mant = c & 7;
    return ex == 0 ? mant * (1.f / 8.f) * (1.f / 64.f) : (1.f + mant * (1.f / 8.f)) * exp2f((float)(ex - 7));
}

// amax[0] = max(amax[0], |w|) over `count` bf16 values (non-negative floats order as their bits).
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_h4_amax(const uint16_t* __restrict__ w, long long count, unsigned* __restrict__ amax) {
    float m = 0.f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (long long)gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(__uint_as_float(((unsigned)w[i]) << 16)));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __shared__ float red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = 0.f;
        for (int i = 0; i < 8; ++i) r = fmaxf(r, red[i]);
        atomicMax(amax, __float_as_uint(r));
    }
}

// One thread a 16-input block of `rows` rows of D bf16: codes [rows][D/2] and e4m3 scales [rows][D/16].
extern "C" __global__ void __launch_bounds__(256) tf_ink2d_h4_quant(const uint16_t* __restrict__ w, int rows, float gs, uint8_t* __restrict__ codes,
                                                                    uint8_t* __restrict__ scales) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)rows * GROUPS) return;
    const uint16_t* src = w + t * GROUP;
    float v[GROUP];
    float bmax = 0.f;
#pragma unroll
    for (int j = 0; j < GROUP; ++j) {
        v[j] = __uint_as_float(((unsigned)src[j]) << 16);
        bmax = fmaxf(bmax, fabsf(v[j]));
    }
    const float want = bmax / 6.f / gs;
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (e4m3Value(mid) < want) lo = mid + 1; else hi = mid;
    }
    const int sc = (lo > 0 && want - e4m3Value(lo - 1) <= e4m3Value(lo) - want) ? lo - 1 : lo;
    scales[t] = (uint8_t)sc;
    const float step = e4m3Value(sc) * gs;
    uint32_t words[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < GROUP; ++j) {
        const float mag = step > 0.f ? fabsf(v[j]) / step : 0.f;
        uint32_t idx = (mag >= 0.25f) + (mag >= 0.75f) + (mag >= 1.25f) + (mag >= 1.75f) + (mag >= 2.5f) + (mag >= 3.5f) + (mag >= 5.0f);
        const uint32_t nib = idx | ((v[j] < 0.f && idx != 0) ? 8u : 0u);
        words[j >> 3] |= nib << (4 * (j & 7));
    }
    *reinterpret_cast<uint2*>(codes + t * (GROUP / 2)) = make_uint2(words[0], words[1]);
}
