// Inkling 2 Mini's NVFP4 routed experts on prompt rows: NVFP4 activation rows times the checkpoint's NVFP4 weights, read in
// place (no repacking), on the block-scaled FP4 mma (sm_121a). A pair tile is one expert's pairs; the weights' per-expert
// outer scale and the activation row's outer scale multiply the sum, as dequant_nvfp4 and nvfp4_act apply them.

#include <cuda_bf16.h>
#include <stdint.h>

#include "mma4.cuh"

namespace ink2p {

// Pulls one contiguous range (16-byte aligned, a multiple of 16 bytes) into L2: whole weight rows reach DRAM as long streams.
__device__ __forceinline__ void prefetchL2(const void* p, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes) : "memory");
}

constexpr int kWarps = 4;
constexpr int kTj = 4;                       // n8 tiles a warp computes
constexpr int kMt = 4;                       // m16 tiles a warp computes
constexpr int kPairs = 16 * kMt;             // pairs of one expert a block holds
constexpr int kBlockCols = kWarps * kTj * 8; // output columns a block computes
constexpr float kNvOuter = 1536.0f;

__device__ __forceinline__ float bf16Round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

template <int kMt, int kTj>
struct Step {
    uint32_t a[kMt][4], sa[kMt];
    uint2 b[kTj];
    uint32_t sb[kTj];
};

// Grid: x = column blocks, y = experts, z = tiles of kPairs of an expert's pairs (it gets at most `rows` pairs).
// Pair p reads activation row p / act_div. DOWN: fp32 out [pairs, n] holds bf16-rounded sums; else bf16 out.
template <int kMt, int kTj, bool DOWN>
__device__ __forceinline__ void grouped(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs,
                                        const float* __restrict__ amax_a, int act_div, const int* __restrict__ offsets,
                                        const int* __restrict__ items, const uint8_t* __restrict__ w,
                                        const uint8_t* __restrict__ ws, const float* __restrict__ amax_w,
                                        const float* __restrict__ denominator, void* __restrict__ out, int N, int K, int pf) {
    const int e = blockIdx.y;
    const int begin = offsets[e];
    constexpr int kPairs = 16 * kMt, kBlockCols = kWarps * kTj * 8;
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
    if (pf) {
        const int tid = threadIdx.x;
        if (tid < kBlockCols) prefetchL2(w + ((size_t)e * N + blockIdx.x * kBlockCols + tid) * (K / 2), (uint32_t)(K / 2));
        if (tid == kWarps * 32 - 1) prefetchL2(ws + ((size_t)e * N + blockIdx.x * kBlockCols) * (K / 16), (uint32_t)(kBlockCols * (K / 16)));
    }

    auto fetch = [&](int kg, Step<kMt, kTj>& f) {
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

    Step<kMt, kTj> st[2];
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


// MXFP8: e4m3 codes with a UE8M0 scale a 32 on both sides, on the block-scaled e4m3 mma (m16n8k32): no outer scale.
__device__ __forceinline__ void mmaMx8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%11, %12}, %13, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
#else
    __trap();
#endif
}

template <int kMt, int kTj>
struct Step8 {
    uint32_t a[kMt][4], sa[kMt];
    uint2 b[kTj];
    uint32_t sb[kTj];
};

// As `grouped` on MXFP8: act codes [rows, K] bytes, exps [rows, K/32], weights [E, N, K] bytes, ws [E, N, K/32].
template <int kMt, int kTj, bool DOWN>
__device__ __forceinline__ void groupedMx(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs, int act_div,
                                          const int* __restrict__ offsets, const int* __restrict__ items,
                                          const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                                          void* __restrict__ out, int N, int K, int pf) {
    const int e = blockIdx.y;
    const int begin = offsets[e];
    constexpr int kPairs = 16 * kMt, kBlockCols = kWarps * kTj * 8;
    const int first = blockIdx.z * kPairs;
    const int cnt = min(offsets[e + 1] - begin - first, kPairs);
    if (cnt <= 0) return;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * kBlockCols + (threadIdx.x >> 5) * kTj * 8;
    const int KG = K / 32;

    const uint8_t* c0[kMt];
    const uint8_t* c1[kMt];
    const uint8_t* sr[kMt];
    bool v0[kMt], v1[kMt];
    int p0[kMt], p1[kMt];
    const bool hi = t & 1;
#pragma unroll
    for (int m = 0; m < kMt; ++m) {
        const int r0 = m * 16 + g, r1 = r0 + 8;
        v0[m] = r0 < cnt;
        v1[m] = r1 < cnt;
        p0[m] = v0[m] ? __ldg(items + begin + first + r0) : 0;
        p1[m] = v1[m] ? __ldg(items + begin + first + r1) : 0;
        const int x0 = p0[m] / act_div, x1 = p1[m] / act_div;
        c0[m] = codes + (size_t)x0 * K;
        c1[m] = codes + (size_t)x1 * K;
        sr[m] = xs + (size_t)(hi ? x1 : x0) * (K / 32);
    }
    const uint8_t* wr[kTj];
    const uint8_t* sw[kTj];
#pragma unroll
    for (int j = 0; j < kTj; ++j) {
        const size_t row = (size_t)e * N + n0 + j * 8 + g;
        wr[j] = w + row * K;
        sw[j] = ws + row * (K / 32);
    }
    if (pf) {
        const int tid = threadIdx.x;
        if (tid < kBlockCols) prefetchL2(w + ((size_t)e * N + blockIdx.x * kBlockCols + tid) * K, (uint32_t)K);
        if (tid == kWarps * 32 - 1) prefetchL2(ws + ((size_t)e * N + blockIdx.x * kBlockCols) * (K / 32), (uint32_t)(kBlockCols * (K / 32)));
    }

    auto fetch = [&](int kg, Step8<kMt, kTj>& f) {
#pragma unroll
        for (int m = 0; m < kMt; ++m) {
            f.a[m][0] = v0[m] ? __ldg(reinterpret_cast<const uint32_t*>(c0[m] + kg * 32 + 4 * t)) : 0u;
            f.a[m][1] = v1[m] ? __ldg(reinterpret_cast<const uint32_t*>(c1[m] + kg * 32 + 4 * t)) : 0u;
            f.a[m][2] = v0[m] ? __ldg(reinterpret_cast<const uint32_t*>(c0[m] + kg * 32 + 16 + 4 * t)) : 0u;
            f.a[m][3] = v1[m] ? __ldg(reinterpret_cast<const uint32_t*>(c1[m] + kg * 32 + 16 + 4 * t)) : 0u;
            f.sa[m] = (hi ? v1[m] : v0[m]) ? (uint32_t)__ldg(sr[m] + kg) : 0u;
        }
#pragma unroll
        for (int j = 0; j < kTj; ++j) {
            f.b[j] = make_uint2(__ldg(reinterpret_cast<const uint32_t*>(wr[j] + kg * 32 + 4 * t)),
                                __ldg(reinterpret_cast<const uint32_t*>(wr[j] + kg * 32 + 16 + 4 * t)));
            f.sb[j] = (uint32_t)__ldg(sw[j] + kg);
        }
    };

    float acc[kMt][kTj][4];
#pragma unroll
    for (int m = 0; m < kMt; ++m)
#pragma unroll
        for (int j = 0; j < kTj; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.0f;

    Step8<kMt, kTj> st[2];
    fetch(0, st[0]);
    for (int kg = 0; kg < KG; kg += 2) {
        if (kg + 1 < KG) fetch(kg + 1, st[1]);
#pragma unroll
        for (int m = 0; m < kMt; ++m)
#pragma unroll
            for (int j = 0; j < kTj; ++j) mmaMx8(acc[m][j], st[0].a[m], st[0].b[j].x, st[0].b[j].y, st[0].sa[m], st[0].sb[j]);
        if (kg + 1 >= KG) break;
        if (kg + 2 < KG) fetch(kg + 2, st[0]);
#pragma unroll
        for (int m = 0; m < kMt; ++m)
#pragma unroll
            for (int j = 0; j < kTj; ++j) mmaMx8(acc[m][j], st[1].a[m], st[1].b[j].x, st[1].b[j].y, st[1].sa[m], st[1].sb[j]);
    }

#pragma unroll
    for (int m = 0; m < kMt; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (!(h ? v1[m] : v0[m])) continue;
            const int p = h ? p1[m] : p0[m];
#pragma unroll
            for (int j = 0; j < kTj; ++j) {
                const int col = n0 + j * 8 + 2 * t;
                const float s0 = acc[m][j][2 * h], s1 = acc[m][j][2 * h + 1];
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

// ---- shared-memory pipelined NVFP4 experts (TF_INK2_PF_SMEM): the same mma, operands and k order as `grouped`, so its sums are
// bit for bit the same; the 16-byte chunks of a k64 step reach the warps through a cp.async ring instead of per-lane global loads.
// Adapted from MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold patch 0015 (Apache-2.0): its staged activation and weight tiles.
namespace ink2p {

__device__ __forceinline__ void cpAsync16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src)
                 : "memory");
}
__device__ __forceinline__ void cpAsync4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src)
                 : "memory");
}
__device__ __forceinline__ void cpCommit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cpWait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ int swz(int r) { return (r >> 2) & 1; }  // 16-byte chunk swap every fourth 32-byte row

template <int kMt, int kTj, int kW, int kSmemStages>
struct StageBuf {
    static constexpr int P = 16 * kMt, C = kW * kTj * 8;
    uint4 a[kSmemStages][P * 2];
    uint32_t as[kSmemStages][P];
    uint4 b[kSmemStages][C * 2];
    uint32_t bs[kSmemStages][C];
};

template <int kMt, int kTj, int kW, bool DOWN, int kSmemStages>
__device__ __forceinline__ void groupedS(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ xs,
                                         const float* __restrict__ amax_a, int act_div, const int* __restrict__ offsets,
                                         const int* __restrict__ items, const uint8_t* __restrict__ w,
                                         const uint8_t* __restrict__ ws, const float* __restrict__ amax_w,
                                         const float* __restrict__ denominator, void* __restrict__ out, int N, int K, int pf) {
    constexpr int P = 16 * kMt, C = kW * kTj * 8, kThreads = kW * 32;
    constexpr int kChunkSlotsA = (2 * P + kThreads - 1) / kThreads, kChunkSlotsB = (2 * C + kThreads - 1) / kThreads;
    constexpr int kScaleSlotsA = (P + kThreads - 1) / kThreads, kScaleSlotsB = (C + kThreads - 1) / kThreads;
    __shared__ StageBuf<kMt, kTj, kW, kSmemStages> sm;
    const int e = blockIdx.y;
    const int begin = offsets[e];
    const int first = blockIdx.z * P;
    const int cnt = min(offsets[e + 1] - begin - first, P);
    if (cnt <= 0) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int nblk = blockIdx.x * C;
    const int KG = K / 64;
    const bool hi = t & 1;

    // Per-thread source rows of the stage copies.
    const uint8_t* asrc[kChunkSlotsA];
    int adst[kChunkSlotsA];
    const uint8_t* assrc[kScaleSlotsA];
#pragma unroll
    for (int i = 0; i < kChunkSlotsA; ++i) {
        const int q = tid + i * kThreads, r = q >> 1, c = q & 1;
        asrc[i] = nullptr;
        adst[i] = 0;
        if (q < 2 * P && r < cnt) {
            const int x = __ldg(items + begin + first + r) / act_div;
            asrc[i] = codes + (size_t)x * (K / 2) + 16 * c;
            adst[i] = r * 2 + (c ^ swz(r));
        }
    }
#pragma unroll
    for (int i = 0; i < kScaleSlotsA; ++i) {
        const int r = tid + i * kThreads;
        assrc[i] = nullptr;
        if (r < P && r < cnt) assrc[i] = xs + (size_t)(__ldg(items + begin + first + r) / act_div) * (K / 16);
    }
    const uint8_t* bsrc[kChunkSlotsB];
    int bdst[kChunkSlotsB];
    const uint8_t* bssrc[kScaleSlotsB];
#pragma unroll
    for (int i = 0; i < kChunkSlotsB; ++i) {
        const int q = tid + i * kThreads, r = q >> 1, c = q & 1;
        bsrc[i] = nullptr;
        bdst[i] = 0;
        if (q < 2 * C) {
            bsrc[i] = w + ((size_t)e * N + nblk + r) * (K / 2) + 16 * c;
            bdst[i] = r * 2 + (c ^ swz(r));
        }
    }
#pragma unroll
    for (int i = 0; i < kScaleSlotsB; ++i) {
        const int r = tid + i * kThreads;
        bssrc[i] = nullptr;
        if (r < C) bssrc[i] = ws + ((size_t)e * N + nblk + r) * (K / 16);
    }

    if (pf) {
        if (tid < C) prefetchL2(w + ((size_t)e * N + nblk + tid) * (K / 2), (uint32_t)(K / 2));
        if (tid == kThreads - 1) prefetchL2(ws + ((size_t)e * N + nblk) * (K / 16), (uint32_t)(C * (K / 16)));
    }
    auto stage = [&](int s, int kg) {
#pragma unroll
        for (int i = 0; i < kChunkSlotsA; ++i)
            if (asrc[i]) cpAsync16(&sm.a[s][adst[i]], asrc[i] + (size_t)kg * 32);
#pragma unroll
        for (int i = 0; i < kScaleSlotsA; ++i)
            if (assrc[i]) cpAsync4(&sm.as[s][tid + i * kThreads], assrc[i] + kg * 4);
#pragma unroll
        for (int i = 0; i < kChunkSlotsB; ++i)
            if (bsrc[i]) cpAsync16(&sm.b[s][bdst[i]], bsrc[i] + (size_t)kg * 32);
#pragma unroll
        for (int i = 0; i < kScaleSlotsB; ++i)
            if (bssrc[i]) cpAsync4(&sm.bs[s][tid + i * kThreads], bssrc[i] + kg * 4);
    };

    float acc[kMt][kTj][4];
#pragma unroll
    for (int m = 0; m < kMt; ++m)
#pragma unroll
        for (int j = 0; j < kTj; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.0f;

#pragma unroll
    for (int s = 0; s < kSmemStages - 1; ++s) {
        if (s < KG) stage(s, s);
        cpCommit();
    }
    for (int k = 0; k < KG; ++k) {
        cpWait<kSmemStages - 2>();
        __syncthreads();
        if (k + kSmemStages - 1 < KG) stage((k + kSmemStages - 1) % kSmemStages, k + kSmemStages - 1);
        cpCommit();
        const int s = k % kSmemStages;
        const uint32_t abase = (uint32_t)__cvta_generic_to_shared(sm.a[s]);
        const uint32_t bbase = (uint32_t)__cvta_generic_to_shared(sm.b[s]);
        uint32_t bf[kTj][2];
        uint32_t sb[kTj];
#pragma unroll
        for (int j = 0; j < kTj; j += 2) {
            // matrices: (tile j, chunk 0), (tile j, chunk 1), (tile j + 1, chunk 0), (tile j + 1, chunk 1)
            const int r = warp * kTj * 8 + (j + ((lane >> 4) & 1)) * 8 + (lane & 7);
            const uint32_t addr = bbase + (uint32_t)((r * 2 + (((lane >> 3) & 1) ^ swz(r))) * 16);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(bf[j][0]), "=r"(bf[j][1]), "=r"(bf[j + 1][0]), "=r"(bf[j + 1][1])
                         : "r"(addr));
        }
#pragma unroll
        for (int j = 0; j < kTj; ++j) sb[j] = sm.bs[s][warp * kTj * 8 + j * 8 + g];
#pragma unroll
        for (int m = 0; m < kMt; ++m) {
            // matrices: (rows 0-7, chunk 0), (rows 8-15, chunk 0), (rows 0-7, chunk 1), (rows 8-15, chunk 1)
            const int ra = m * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
            const uint32_t addr = abase + (uint32_t)((ra * 2 + ((lane >> 4) ^ swz(ra))) * 16);
            uint32_t a[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                         : "r"(addr));
            const uint32_t sa = sm.as[s][m * 16 + g + (hi ? 8 : 0)];
#pragma unroll
            for (int j = 0; j < kTj; ++j) mma4::mma_fp4(acc[m][j], a, bf[j][0], bf[j][1], sa, sb[j]);
        }
    }
    cpWait<0>();

    const int n0 = nblk + warp * kTj * 8;
    const float outer_w = __fdiv_rn(__ldg(amax_w + e), __ldg(denominator));
#pragma unroll
    for (int m = 0; m < kMt; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = m * 16 + g + 8 * h;
            if (r >= cnt) continue;
            const int p = __ldg(items + begin + first + r);
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

#define INK2P_NVFP4S_ENTRY(MT, TJ, NW, ST, SUF)                                                                                   \
    extern "C" __global__ void __launch_bounds__(NW * 32)                                                     \
        tf_ink2p_gu_nvfp4##SUF(const uint8_t* codes, const uint8_t* xs, const float* amax_a, int act_div,                 \
                               const int* offsets, const int* items, const uint8_t* w, const uint8_t* ws,                 \
                               const float* amax_w, const float* denominator, __nv_bfloat16* out, int N, int K, int pf) {         \
        ink2p::groupedS<MT, TJ, NW, false, ST>(codes, xs, amax_a, act_div, offsets, items, w, ws, amax_w, denominator, out, N, K, pf); \
    }                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(NW * 32)                                                     \
        tf_ink2p_down_nvfp4##SUF(const uint8_t* codes, const uint8_t* xs, const float* amax_a, int act_div,               \
                                 const int* offsets, const int* items, const uint8_t* w, const uint8_t* ws,               \
                                 const float* amax_w, const float* denominator, float* out, int N, int K, int pf) {               \
        ink2p::groupedS<MT, TJ, NW, true, ST>(codes, xs, amax_a, act_div, offsets, items, w, ws, amax_w, denominator, out, N, K, pf);  \
    }

INK2P_NVFP4S_ENTRY(4, 4, 4, 6, _s6)
INK2P_NVFP4S_ENTRY(4, 4, 4, 4, _s4)
INK2P_NVFP4S_ENTRY(4, 4, 4, 3, _s3)
INK2P_NVFP4S_ENTRY(8, 2, 4, 6, _m8t2_s6)
INK2P_NVFP4S_ENTRY(8, 2, 4, 4, _m8t2_s4)
INK2P_NVFP4S_ENTRY(8, 2, 8, 4, _m8t2_w8s4)
INK2P_NVFP4S_ENTRY(8, 2, 8, 5, _m8t2_w8s5)
INK2P_NVFP4S_ENTRY(8, 2, 8, 3, _m8t2_w8s3)

#define INK2P_NVFP4_ENTRY(MT, TJ, SUF)                                                                                    \
    extern "C" __global__ void __launch_bounds__(ink2p::kWarps * 32)                                                     \
        tf_ink2p_gu_nvfp4##SUF(const uint8_t* codes, const uint8_t* xs, const float* amax_a, int act_div,                 \
                               const int* offsets, const int* items, const uint8_t* w, const uint8_t* ws,                 \
                               const float* amax_w, const float* denominator, __nv_bfloat16* out, int N, int K, int pf) {         \
        ink2p::grouped<MT, TJ, false>(codes, xs, amax_a, act_div, offsets, items, w, ws, amax_w, denominator, out, N, K, pf); \
    }                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(ink2p::kWarps * 32)                                                     \
        tf_ink2p_down_nvfp4##SUF(const uint8_t* codes, const uint8_t* xs, const float* amax_a, int act_div,               \
                                 const int* offsets, const int* items, const uint8_t* w, const uint8_t* ws,               \
                                 const float* amax_w, const float* denominator, float* out, int N, int K, int pf) {               \
        ink2p::grouped<MT, TJ, true>(codes, xs, amax_a, act_div, offsets, items, w, ws, amax_w, denominator, out, N, K, pf);  \
    }

INK2P_NVFP4_ENTRY(4, 4, )
INK2P_NVFP4_ENTRY(8, 2, _m8t2)
INK2P_NVFP4_ENTRY(8, 4, _m8t4)
INK2P_NVFP4_ENTRY(2, 4, _m2t4)
INK2P_NVFP4_ENTRY(16, 1, _m16t1)
INK2P_NVFP4_ENTRY(12, 2, _m12t2)

// ---- bf16 rows times bf16 weight rows on the bf16 mma: out[r, n] = x[r] . w[n], float32 ----
// The sum of a column runs on the tensor cores in k16 steps, so it differs from tf_dense_rows in order only.
namespace ink2p {

constexpr int kDenseWarps = 4;
constexpr int kDenseMt = 4;   // m16 tiles a warp computes
constexpr int kDenseNt = 4;   // n8 tiles a warp computes
constexpr int kDenseRows = 16 * kDenseMt;
constexpr int kDenseCols = kDenseWarps * kDenseNt * 8;

__device__ __forceinline__ void mmaBf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

struct DenseStep {
    uint32_t a[kDenseMt][4];
    uint32_t b[kDenseNt][2];
};

}  // namespace ink2p

// Grid: x = tiles of kDenseCols columns, y = tiles of kDenseRows rows. x [m, k] bf16 with row stride x_stride (elements),
// w [n, k] bf16, out [m, n] with row stride out_stride: float32, or bf16 rounded once (BF16OUT). k % 16 == 0 and n % 8 == 0.
template <bool BF16OUT>
__device__ __forceinline__ void denseRows(const __nv_bfloat16* __restrict__ x, long long x_stride, const __nv_bfloat16* __restrict__ w,
                                          void* __restrict__ out_v, long long out_stride, int m, int n, int k) {
    using namespace ink2p;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int col0 = blockIdx.x * kDenseCols + (threadIdx.x >> 5) * kDenseNt * 8;
    const int row0 = blockIdx.y * kDenseRows;
    if (col0 >= n) return;
    const uint32_t* xr[kDenseMt][2];
    const uint32_t* wr[kDenseNt];
#pragma unroll
    for (int mt = 0; mt < kDenseMt; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = min(row0 + mt * 16 + g + 8 * h, m - 1);
            xr[mt][h] = reinterpret_cast<const uint32_t*>(x + (size_t)r * x_stride) + t;
        }
#pragma unroll
    for (int j = 0; j < kDenseNt; ++j) wr[j] = reinterpret_cast<const uint32_t*>(w + (size_t)min(col0 + j * 8 + g, n - 1) * k) + t;

    auto fetch = [&](int kk, DenseStep& f) {  // k16 step kk: words kk * 8 + t and kk * 8 + 4 + t of each row
#pragma unroll
        for (int mt = 0; mt < kDenseMt; ++mt) {
            f.a[mt][0] = __ldg(xr[mt][0] + kk * 8);
            f.a[mt][1] = __ldg(xr[mt][1] + kk * 8);
            f.a[mt][2] = __ldg(xr[mt][0] + kk * 8 + 4);
            f.a[mt][3] = __ldg(xr[mt][1] + kk * 8 + 4);
        }
#pragma unroll
        for (int j = 0; j < kDenseNt; ++j) {
            f.b[j][0] = __ldg(wr[j] + kk * 8);
            f.b[j][1] = __ldg(wr[j] + kk * 8 + 4);
        }
    };
    float acc[kDenseMt][kDenseNt][4];
#pragma unroll
    for (int mt = 0; mt < kDenseMt; ++mt)
#pragma unroll
        for (int j = 0; j < kDenseNt; ++j) acc[mt][j][0] = acc[mt][j][1] = acc[mt][j][2] = acc[mt][j][3] = 0.0f;

    const int steps = k / 16;
    DenseStep st[2];
    fetch(0, st[0]);
    for (int kk = 0; kk < steps; kk += 2) {
        if (kk + 1 < steps) fetch(kk + 1, st[1]);
#pragma unroll
        for (int mt = 0; mt < kDenseMt; ++mt)
#pragma unroll
            for (int j = 0; j < kDenseNt; ++j) mmaBf16(acc[mt][j], st[0].a[mt], st[0].b[j][0], st[0].b[j][1]);
        if (kk + 1 >= steps) break;
        if (kk + 2 < steps) fetch(kk + 2, st[0]);
#pragma unroll
        for (int mt = 0; mt < kDenseMt; ++mt)
#pragma unroll
            for (int j = 0; j < kDenseNt; ++j) mmaBf16(acc[mt][j], st[1].a[mt], st[1].b[j][0], st[1].b[j][1]);
    }
#pragma unroll
    for (int mt = 0; mt < kDenseMt; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = row0 + mt * 16 + g + 8 * h;
            if (r >= m) continue;
#pragma unroll
            for (int j = 0; j < kDenseNt; ++j) {
                const int col = col0 + j * 8 + 2 * t;
                if (col >= n) continue;
                if constexpr (BF16OUT) {
                    *reinterpret_cast<__nv_bfloat162*>(static_cast<__nv_bfloat16*>(out_v) + (size_t)r * out_stride + col) =
                        __floats2bfloat162_rn(acc[mt][j][2 * h], acc[mt][j][2 * h + 1]);
                } else {
                    *reinterpret_cast<float2*>(static_cast<float*>(out_v) + (size_t)r * out_stride + col) = make_float2(acc[mt][j][2 * h], acc[mt][j][2 * h + 1]);
                }
            }
        }
}

extern "C" __global__ void __launch_bounds__(ink2p::kDenseWarps * 32)
    tf_ink2p_dense_rows(const __nv_bfloat16* __restrict__ x, long long x_stride, const __nv_bfloat16* __restrict__ w,
                        float* __restrict__ out, long long out_stride, int m, int n, int k) {
    denseRows<false>(x, x_stride, w, out, out_stride, m, n, k);
}


extern "C" __global__ void __launch_bounds__(ink2p::kDenseWarps * 32)
    tf_ink2p_dense_rows_bf16(const __nv_bfloat16* __restrict__ x, long long x_stride, const __nv_bfloat16* __restrict__ w,
                             __nv_bfloat16* __restrict__ out, long long out_stride, int m, int n, int k) {
    denseRows<true>(x, x_stride, w, out, out_stride, m, n, k);
}


#define INK2P_MX8_ENTRY(MT, TJ, SUF)                                                                                  \
    extern "C" __global__ void __launch_bounds__(ink2p::kWarps * 32)                                                 \
        tf_ink2p_gu_mxfp8##SUF(const uint8_t* codes, const uint8_t* xs, int act_div, const int* offsets,              \
                               const int* items, const uint8_t* w, const uint8_t* ws, __nv_bfloat16* out, int N, int K, int pf) { \
        ink2p::groupedMx<MT, TJ, false>(codes, xs, act_div, offsets, items, w, ws, out, N, K, pf);                        \
    }                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(ink2p::kWarps * 32)                                                 \
        tf_ink2p_down_mxfp8##SUF(const uint8_t* codes, const uint8_t* xs, int act_div, const int* offsets,            \
                                 const int* items, const uint8_t* w, const uint8_t* ws, float* out, int N, int K, int pf) {   \
        ink2p::groupedMx<MT, TJ, true>(codes, xs, act_div, offsets, items, w, ws, out, N, K, pf);                         \
    }

INK2P_MX8_ENTRY(4, 4, )
INK2P_MX8_ENTRY(8, 2, _m8t2)
INK2P_MX8_ENTRY(8, 4, _m8t4)
