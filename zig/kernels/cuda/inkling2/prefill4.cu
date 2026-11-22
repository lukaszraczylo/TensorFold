// Inkling 2 Mini's NVFP4 routed experts on prompt rows: NVFP4 activation rows times the checkpoint's NVFP4 weights, read in
// place (no repacking), on the block-scaled FP4 mma (sm_121a). A pair tile is one expert's pairs; the weights' per-expert
// outer scale and the activation row's outer scale multiply the sum, as dequant_nvfp4 and nvfp4_act apply them.

#include <cuda_bf16.h>
#include <stdint.h>

#include "mma4.cuh"

namespace ink2p {

constexpr int kWarps = 4;
constexpr int kTj = 4;                       // n8 tiles a warp computes
constexpr int kMt = 2;                       // m16 tiles a warp computes
constexpr int kPairs = 16 * kMt;             // pairs of one expert a block holds
constexpr int kBlockCols = kWarps * kTj * 8; // output columns a block computes
constexpr float kNvOuter = 1536.0f;

__device__ __forceinline__ float bf16Round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

struct Step {
    uint32_t a[kMt][4], sa[kMt];
    uint2 b[kTj];
    uint32_t sb[kTj];
};

// Grid: x = column blocks, y = experts, z = tiles of kPairs of an expert's pairs (it gets at most `rows` pairs).
// Pair p reads activation row p / act_div. DOWN: fp32 out [pairs, n] holds bf16-rounded sums; else bf16 out.
template <bool DOWN>
__device__ __forceinline__ void grouped(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs,
                                        const float* __restrict__ amax_a, int act_div, const int* __restrict__ offsets,
                                        const int* __restrict__ items, const uint8_t* __restrict__ w,
                                        const uint8_t* __restrict__ ws, const float* __restrict__ amax_w,
                                        const float* __restrict__ denominator, void* __restrict__ out, int N, int K) {
    const int e = blockIdx.y;
    const int begin = offsets[e];
    const int first = blockIdx.z * kPairs;
    const int cnt = min(offsets[e + 1] - begin - first, kPairs);
    if (cnt <= 0) return;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * kBlockCols + (threadIdx.x >> 5) * kTj * 8;
    const int KG = K / 64;

    const uint8_t* c0[kMt];
    const uint8_t* c1[kMt];
    const uint8_t* sr[kMt];
    bool v0[kMt], v1[kMt];
    int p0[kMt], p1[kMt];
    const bool hi = t & 1;  // lanes t 0 and 1 give rows g and g + 8's four block scales; t 2, 3 are not read
#pragma unroll
    for (int m = 0; m < kMt; ++m) {
        const int r0 = m * 16 + g, r1 = r0 + 8;
        v0[m] = r0 < cnt;
        v1[m] = r1 < cnt;
        p0[m] = v0[m] ? __ldg(items + begin + first + r0) : 0;
        p1[m] = v1[m] ? __ldg(items + begin + first + r1) : 0;
        const int x0 = p0[m] / act_div, x1 = p1[m] / act_div;
        c0[m] = codes + (size_t)x0 * (K / 2);
        c1[m] = codes + (size_t)x1 * (K / 2);
        sr[m] = xs + (size_t)(hi ? x1 : x0) * (K / 16);
    }
    const uint8_t* wr[kTj];
    const uint8_t* sw[kTj];
#pragma unroll
    for (int j = 0; j < kTj; ++j) {
        const size_t row = (size_t)e * N + n0 + j * 8 + g;
        wr[j] = w + row * (K / 2);
        sw[j] = ws + row * (K / 16);
    }

    auto fetch = [&](int kg, Step& f) {
#pragma unroll
        for (int m = 0; m < kMt; ++m) {
            f.a[m][0] = v0[m] ? __ldg(reinterpret_cast<const uint32_t*>(c0[m] + kg * 32 + 4 * t)) : 0u;
            f.a[m][1] = v1[m] ? __ldg(reinterpret_cast<const uint32_t*>(c1[m] + kg * 32 + 4 * t)) : 0u;
            f.a[m][2] = v0[m] ? __ldg(reinterpret_cast<const uint32_t*>(c0[m] + kg * 32 + 16 + 4 * t)) : 0u;
            f.a[m][3] = v1[m] ? __ldg(reinterpret_cast<const uint32_t*>(c1[m] + kg * 32 + 16 + 4 * t)) : 0u;
            f.sa[m] = (hi ? v1[m] : v0[m]) ? __ldg(reinterpret_cast<const uint32_t*>(sr[m] + kg * 4)) : 0u;
        }
#pragma unroll
        for (int j = 0; j < kTj; ++j) {
            f.b[j] = make_uint2(__ldg(reinterpret_cast<const uint32_t*>(wr[j] + kg * 32 + 4 * t)),
                                __ldg(reinterpret_cast<const uint32_t*>(wr[j] + kg * 32 + 16 + 4 * t)));
            f.sb[j] = __ldg(reinterpret_cast<const uint32_t*>(sw[j] + kg * 4));
        }
    };

    float acc[kMt][kTj][4];
#pragma unroll
    for (int m = 0; m < kMt; ++m)
#pragma unroll
        for (int j = 0; j < kTj; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.0f;

    Step st[2];
    fetch(0, st[0]);
    for (int kg = 0; kg < KG; kg += 2) {
        if (kg + 1 < KG) fetch(kg + 1, st[1]);
#pragma unroll
        for (int m = 0; m < kMt; ++m)
#pragma unroll
            for (int j = 0; j < kTj; ++j) mma4::mma_fp4(acc[m][j], st[0].a[m], st[0].b[j].x, st[0].b[j].y, st[0].sa[m], st[0].sb[j]);
        if (kg + 1 >= KG) break;
        if (kg + 2 < KG) fetch(kg + 2, st[0]);
#pragma unroll
        for (int m = 0; m < kMt; ++m)
#pragma unroll
            for (int j = 0; j < kTj; ++j) mma4::mma_fp4(acc[m][j], st[1].a[m], st[1].b[j].x, st[1].b[j].y, st[1].sa[m], st[1].sb[j]);
    }

    const float outer_w = __fdiv_rn(__ldg(amax_w + e), __ldg(denominator));
#pragma unroll
    for (int m = 0; m < kMt; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (!(h ? v1[m] : v0[m])) continue;
            const int p = h ? p1[m] : p0[m];
            const float outer = __fdiv_rn(__ldg(amax_a + p / act_div), kNvOuter);
#pragma unroll
            for (int j = 0; j < kTj; ++j) {
                const int col = n0 + j * 8 + 2 * t;
                const float s0 = __fmul_rn(__fmul_rn(acc[m][j][2 * h], outer_w), outer);
                const float s1 = __fmul_rn(__fmul_rn(acc[m][j][2 * h + 1], outer_w), outer);
                if constexpr (DOWN) {
                    *reinterpret_cast<float2*>(static_cast<float*>(out) + (size_t)p * N + col) = make_float2(bf16Round(s0), bf16Round(s1));
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(out) + (size_t)p * N + col) = __floats2bfloat162_rn(s0, s1);
                }
            }
        }
    }
}

}  // namespace ink2p

extern "C" __global__ void __launch_bounds__(ink2p::kWarps * 32)
    tf_ink2p_gu_nvfp4(const uint8_t* codes, const uint8_t* xs, const float* amax_a, int act_div, const int* offsets,
                      const int* items, const uint8_t* w, const uint8_t* ws, const float* amax_w, const float* denominator,
                      __nv_bfloat16* out, int N, int K) {
    ink2p::grouped<false>(codes, xs, amax_a, act_div, offsets, items, w, ws, amax_w, denominator, out, N, K);
}

extern "C" __global__ void __launch_bounds__(ink2p::kWarps * 32)
    tf_ink2p_down_nvfp4(const uint8_t* codes, const uint8_t* xs, const float* amax_a, int act_div, const int* offsets,
                        const int* items, const uint8_t* w, const uint8_t* ws, const float* amax_w,
                        const float* denominator, float* out, int N, int K) {
    ink2p::grouped<true>(codes, xs, amax_a, act_div, offsets, items, w, ws, amax_w, denominator, out, N, K);
}
