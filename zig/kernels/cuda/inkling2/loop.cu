// Inkling 2 Mini's looped-engine kernels: the check tool's smoke kernel and the expert-input activation quantisers.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" __global__ void tf_ink2l_fill(uint32_t* out, int n, uint32_t v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = v;
}

// Each statement follows ref/quant.py's line, one IEEE operation at a time: no fused multiply-adds, no fast division.
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kNvBlock = 16;
constexpr int kMxBlock = 32;
constexpr float kFp32Max = 3.4028234663852886e38f;
constexpr float kE4m3Max = 448.0f;
constexpr float kNvOuter = 1536.0f;
constexpr float kInvNvOuter = (float)(1.0 / 1536.0);
constexpr float kInvE4m3Max = (float)(1.0 / 448.0);
constexpr int kUe8m0Bias = 127;
constexpr int kUe8m0Max = 254;
constexpr uint32_t kSubnormalScale = 0x00400000u;  // 2^-127, the scale of biased exponent 0

__device__ __forceinline__ float clampMax(float x, float hi) { return x > hi ? hi : x; }

// torch rounds a e4m3 cast to nearest even; the caller clamps first, so satfinite never fires on a finite input.
__device__ __forceinline__ float e4m3Round(float x) {
    const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3)));
}

// ref.quant.e2m1_rne_sat: ties 0.25, 1.25, 2.5, 5.0 go down and 0.75, 1.75, 3.5 go up.
__device__ __forceinline__ float e2m1RneSat(float x) {
    const float a = fabsf(x);
    const int code = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
    const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return copysignf(grid[code], x);
}

// ref.quant._pairwise_sum over 16 values: adjacent pairs, level by level.
__device__ __forceinline__ float pairwise16(float* e) {
    #pragma unroll
    for (int n = kNvBlock; n > 1; n >>= 1) {
        #pragma unroll
        for (int i = 0; i < n / 2; ++i) e[i] = __fadd_rn(e[2 * i], e[2 * i + 1]);
    }
    return e[0];
}

struct Candidate {
    float values[kNvBlock];
    float scale;
    float err;
};

// ref.quant._nvfp4_candidate: `factor` is the float32 of 1 / divisor, as the reference builds it.
__device__ __forceinline__ void nvfp4Candidate(
    const float* blocks, float block_max, float encoder, float decoder, float err_outer, float factor, Candidate& c) {
    const float scale_raw = clampMax(__fmul_rn(block_max, __fmul_rn(encoder, factor)), kE4m3Max);
    c.scale = e4m3Round(scale_raw);
    const float inverse = clampMax(__fdiv_rn(1.0f, __fmul_rn(c.scale, decoder)), kFp32Max);
    float diff[kNvBlock];
    #pragma unroll
    for (int i = 0; i < kNvBlock; ++i) {
        c.values[i] = e2m1RneSat(__fmul_rn(blocks[i], inverse));
        const float rounded = __fmul_rn(c.values[i], c.scale);
        diff[i] = fabsf(__fsub_rn(__fmul_rn(rounded, err_outer), blocks[i]));
    }
    c.err = pairwise16(diff);
}

__device__ float blockMax(float v, float* red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = red[0];
    for (int w = 1; w < kWarps; ++w) r = fmaxf(r, red[w]);
    __syncthreads();
    return r;
}

// One block a row. out = dequantised float32, the array ref.quant.nvfp4_act returns.
extern "C" __global__ void tf_ink2l_actq_nvfp4(const __nv_bfloat16* x, float* out, int rows, int k) {
    __shared__ float red[kWarps];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* xr = x + (size_t)row * k;
    float* outr = out + (size_t)row * k;

    float m = 0.0f;
    for (int i = threadIdx.x; i < k; i += kThreads) m = fmaxf(m, fabsf(__bfloat162float(xr[i])));
    const float amax = blockMax(m, red);

    // torch evaluates `float / tensor` as reciprocal(tensor) * float, two roundings, so the encoder does too.
    const float encoder = clampMax(__fmul_rn(__fdiv_rn(1.0f, amax), kNvOuter), kFp32Max);
    const float decoder = __fdiv_rn(1.0f, encoder);
    const float err_outer = __fmul_rn(amax, kInvNvOuter);
    const float outer = __fdiv_rn(amax, kNvOuter);
    const float factor6 = (float)(1.0 / 6.0);
    const float factor4 = (float)(1.0 / 4.0);

    for (int b = threadIdx.x; b < k / kNvBlock; b += kThreads) {
        float blocks[kNvBlock];
        float block_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < kNvBlock; ++i) {
            blocks[i] = __bfloat162float(xr[b * kNvBlock + i]);
            block_max = fmaxf(block_max, fabsf(blocks[i]));
        }
        Candidate c6, c4;
        nvfp4Candidate(blocks, block_max, encoder, decoder, err_outer, factor6, c6);
        nvfp4Candidate(blocks, block_max, encoder, decoder, err_outer, factor4, c4);
        const Candidate& w = (c4.err < c6.err) ? c4 : c6;
        #pragma unroll
        for (int i = 0; i < kNvBlock; ++i) outr[b * kNvBlock + i] = __fmul_rn(__fmul_rn(w.values[i], w.scale), outer);
    }
}

// The UE8M0 scale of one 32-group: the power of two at or above raw, as ref.quant.mxfp8_act builds it.
__device__ __forceinline__ float ue8m0Scale(float raw) {
    if (raw != raw) return __int_as_float(0x7fc00000);
    int biased;
    if (raw == 0.0f) {
        biased = 0;
    } else if (isinf(raw)) {
        biased = kUe8m0Max;
    } else {
        int e;
        const float mantissa = frexpf(raw, &e);
        const int ceil_exp = (mantissa == 0.5f) ? e - 1 : e;
        biased = min(max(ceil_exp + kUe8m0Bias, 0), kUe8m0Max);
    }
    return __uint_as_float(biased == 0 ? kSubnormalScale : (uint32_t)biased << 23);
}

extern "C" __global__ void tf_ink2l_actq_mxfp8(const __nv_bfloat16* x, float* out, int rows, int k) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* xr = x + (size_t)row * k;
    float* outr = out + (size_t)row * k;
    for (int g = threadIdx.x; g < k / kMxBlock; g += kThreads) {
        float v[kMxBlock];
        float gmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < kMxBlock; ++i) {
            v[i] = __bfloat162float(xr[g * kMxBlock + i]);
            gmax = fmaxf(gmax, fabsf(v[i]));
        }
        const float scale = ue8m0Scale(__fmul_rn(gmax, kInvE4m3Max));
        #pragma unroll
        for (int i = 0; i < kMxBlock; ++i) {
            const float q = fminf(fmaxf(__fdiv_rn(v[i], scale), -kE4m3Max), kE4m3Max);
            outr[g * kMxBlock + i] = __fmul_rn(e4m3Round(q), scale);
        }
    }
}
