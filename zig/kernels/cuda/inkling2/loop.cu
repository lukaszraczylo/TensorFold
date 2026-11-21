// Inkling 2 Mini's looped-engine kernels: the check tool's smoke kernel, the activation quantisers and the expert decode path.
// Accumulation order of every output dot (the expert GEMVs here; the prompt path must match): lane l of a warp adds its
// elements k = l, l + 32, l + 64, ... of the K axis in increasing k with fmaf, then the xor-shuffle tree 16, 8, 4, 2, 1.
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
constexpr int kUe8m0Nan = 255;
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

// One block a row. out = dequantised float32, the array ref.quant.nvfp4_act returns. values, scales and
// amax_out (each may be null) are what ref.quant.nvfp4_codes returns: e2m1 values [rows, k], e4m3 block
// scales [rows, k/16] and the row amax [rows], all float32.
__device__ void actqNvfp4Row(
    const __nv_bfloat16* x, float* out, float* values, float* scales, float* amax_out, int row, int k) {
    __shared__ float red[kWarps];
    const __nv_bfloat16* xr = x + (size_t)row * k;
    float* outr = out + (size_t)row * k;
    if (amax_out == nullptr) values = scales = nullptr;

    float m = 0.0f;
    for (int i = threadIdx.x; i < k; i += kThreads) m = fmaxf(m, fabsf(__bfloat162float(xr[i])));
    const float amax = blockMax(m, red);
    if (amax_out != nullptr && threadIdx.x == 0) amax_out[row] = amax;

    const float encoder = clampMax(__fdiv_rn(kNvOuter, amax), kFp32Max);
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
        if (values != nullptr) {
            scales[(size_t)row * (k / kNvBlock) + b] = w.scale;
            #pragma unroll
            for (int i = 0; i < kNvBlock; ++i) values[(size_t)row * k + b * kNvBlock + i] = w.values[i];
        }
    }
}

extern "C" __global__ void tf_ink2l_actq_nvfp4(const __nv_bfloat16* x, float* out, int rows, int k) {
    if (blockIdx.x < rows) actqNvfp4Row(x, out, nullptr, nullptr, nullptr, blockIdx.x, k);
}

extern "C" __global__ void tf_ink2l_actq_nvfp4_codes(
    const __nv_bfloat16* x, float* out, float* values, float* scales, float* amax, int rows, int k) {
    if (blockIdx.x < rows) actqNvfp4Row(x, out, values, scales, amax, blockIdx.x, k);
}

// The biased UE8M0 exponent of one 32-group: the power of two at or above raw, as ref.quant.mxfp8_codes builds it.
__device__ __forceinline__ int ue8m0Biased(float raw) {
    int biased;
    if (raw != raw) {
        biased = kUe8m0Nan;
    } else if (raw == 0.0f) {
        biased = 0;
    } else if (isinf(raw)) {
        biased = kUe8m0Max;
    } else {
        int e;
        const float mantissa = frexpf(raw, &e);
        const int ceil_exp = (mantissa == 0.5f) ? e - 1 : e;
        biased = min(max(ceil_exp + kUe8m0Bias, 0), kUe8m0Max);
    }
    return biased;
}

__device__ __forceinline__ float ue8m0Value(int biased) {
    if (biased == kUe8m0Nan) return __int_as_float(0x7fc00000);
    return __uint_as_float(biased == 0 ? kSubnormalScale : (uint32_t)biased << 23);
}

// values (float32 [rows, k]) and exps (uint8 [rows, k/32]) may be null: ref.quant.mxfp8_codes' outputs.
__device__ void actqMxfp8Row(
    const __nv_bfloat16* x, float* out, float* values, uint8_t* exps, int row, int k) {
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
        const int biased = ue8m0Biased(__fmul_rn(gmax, kInvE4m3Max));
        const float scale = ue8m0Value(biased);
        if (exps != nullptr) exps[(size_t)row * (k / kMxBlock) + g] = (uint8_t)biased;
        #pragma unroll
        for (int i = 0; i < kMxBlock; ++i) {
            const float q = fminf(fmaxf(__fdiv_rn(v[i], scale), -kE4m3Max), kE4m3Max);
            const float qv = e4m3Round(q);
            outr[g * kMxBlock + i] = __fmul_rn(qv, scale);
            if (values != nullptr) values[(size_t)row * k + g * kMxBlock + i] = qv;
        }
    }
}

extern "C" __global__ void tf_ink2l_actq_mxfp8(const __nv_bfloat16* x, float* out, int rows, int k) {
    if (blockIdx.x < rows) actqMxfp8Row(x, out, nullptr, nullptr, blockIdx.x, k);
}

extern "C" __global__ void tf_ink2l_actq_mxfp8_codes(
    const __nv_bfloat16* x, float* out, float* values, uint8_t* exps, int rows, int k) {
    if (blockIdx.x < rows) actqMxfp8Row(x, out, values, exps, blockIdx.x, k);
}

// ---- Expert decode path: GEMV on the stored weights, softcapped SwiGLU, combine ----
// A stage's device template takes the stored format as a WeightFmt; the extern "C" wrappers are the only per-format code.
constexpr int kExWarps = kThreads / 32;  // output columns a block computes
constexpr int kSlots = 8;
constexpr uint32_t kFloatInf = 0x7f800000u;

__constant__ float kE2m1Lut[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float bf16Round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float e4m3ToFloat(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}

// ref.weights.dequant_nvfp4: bf16((lut[code] * f32(block scale)) * outer), outer = amax[e] / denominator, low nibble first.
struct Nvfp4Fmt {
    static constexpr int kPack = 2;  // elements a stored byte holds
    static constexpr int kBlock = kNvBlock;
    __device__ static float expertOuter(const float* amax, const float* denominator, int e) {
        return __fdiv_rn(amax[e], denominator[0]);
    }
    __device__ static float dequant(const uint8_t* w, const uint8_t* scale, int k, float outer) {
        const int code = (w[k >> 1] >> ((k & 1) * 4)) & 0xF;
        return bf16Round(__fmul_rn(__fmul_rn(kE2m1Lut[code], e4m3ToFloat(scale[k / kNvBlock])), outer));
    }
};

// ref.weights.dequant_mxfp8: bf16(f32(e4m3) * 2^(ue8m0 - 127)), no outer scale.
struct Mxfp8Fmt {
    static constexpr int kPack = 1;
    static constexpr int kBlock = kMxBlock;
    __device__ static float expertOuter(const float*, const float*, int) { return 1.0f; }
    __device__ static float dequant(const uint8_t* w, const uint8_t* scale, int k, float) {
        const int biased = scale[k / kMxBlock];
        const float exp2 = biased == kUe8m0Nan ? __uint_as_float(kFloatInf) : ue8m0Value(biased);
        return bf16Round(__fmul_rn(e4m3ToFloat(w[k]), exp2));
    }
};

// One output column n of expert e against the activation row x: the warp's dot product, valid in every lane.
template <class Fmt>
__device__ float expertColumn(const float* x, int e, int n, const uint8_t* w, const uint8_t* scale, const float* amax,
                              const float* denominator, int n_dim, int k_dim) {
    const size_t row = (size_t)e * n_dim + n;
    const uint8_t* wr = w + row * (k_dim / Fmt::kPack);
    const uint8_t* sr = scale + row * (k_dim / Fmt::kBlock);
    const float outer = Fmt::expertOuter(amax, denominator, e);
    float acc = 0.0f;
    for (int k = threadIdx.x & 31; k < k_dim; k += 32) acc = __fmaf_rn(x[k], Fmt::dequant(wr, sr, k, outer), acc);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    return acc;
}

// Grid: x = (row, slot) pairs, y = tiles of kExWarps columns. act float [rows, k]; gu bf16 [rows * 8, n].
template <class Fmt>
__device__ void exGateUp(const float* act, const int* pick, const uint8_t* w, const uint8_t* scale, const float* amax,
                         const float* denominator, __nv_bfloat16* gu, int pairs, int n_dim, int k_dim) {
    const int p = blockIdx.x;
    const int n = blockIdx.y * kExWarps + (threadIdx.x >> 5);
    if (p >= pairs || n >= n_dim) return;
    const float sum = expertColumn<Fmt>(act + (size_t)(p / kSlots) * k_dim, pick[p], n, w, scale, amax, denominator, n_dim, k_dim);
    if ((threadIdx.x & 31) == 0) gu[(size_t)p * n_dim + n] = __float2bfloat16_rn(sum);
}

// act float [rows * 8, k] (the quantised SwiGLU output); dn float [rows * 8, n] holds the bf16-rounded value.
template <class Fmt>
__device__ void exDown(const float* act, const int* pick, const uint8_t* w, const uint8_t* scale, const float* amax,
                       const float* denominator, float* dn, int pairs, int n_dim, int k_dim) {
    const int p = blockIdx.x;
    const int n = blockIdx.y * kExWarps + (threadIdx.x >> 5);
    if (p >= pairs || n >= n_dim) return;
    const float sum = expertColumn<Fmt>(act + (size_t)p * k_dim, pick[p], n, w, scale, amax, denominator, n_dim, k_dim);
    if ((threadIdx.x & 31) == 0) dn[(size_t)p * n_dim + n] = bf16Round(sum);
}

extern "C" __global__ void tf_ink2l_ex_gu_nvfp4(const float* act, const int* pick, const uint8_t* w, const uint8_t* scale,
                                                const float* amax, const float* denominator, __nv_bfloat16* gu, int pairs,
                                                int n_dim, int k_dim) {
    exGateUp<Nvfp4Fmt>(act, pick, w, scale, amax, denominator, gu, pairs, n_dim, k_dim);
}

extern "C" __global__ void tf_ink2l_ex_gu_mxfp8(const float* act, const int* pick, const uint8_t* w, const uint8_t* scale,
                                                const float* amax, const float* denominator, __nv_bfloat16* gu, int pairs,
                                                int n_dim, int k_dim) {
    exGateUp<Mxfp8Fmt>(act, pick, w, scale, amax, denominator, gu, pairs, n_dim, k_dim);
}

extern "C" __global__ void tf_ink2l_ex_down_nvfp4(const float* act, const int* pick, const uint8_t* w, const uint8_t* scale,
                                                  const float* amax, const float* denominator, float* dn, int pairs,
                                                  int n_dim, int k_dim) {
    exDown<Nvfp4Fmt>(act, pick, w, scale, amax, denominator, dn, pairs, n_dim, k_dim);
}

extern "C" __global__ void tf_ink2l_ex_down_mxfp8(const float* act, const int* pick, const uint8_t* w, const uint8_t* scale,
                                                  const float* amax, const float* denominator, float* dn, int pairs,
                                                  int n_dim, int k_dim) {
    exDown<Mxfp8Fmt>(act, pick, w, scale, amax, denominator, dn, pairs, n_dim, k_dim);
}

__device__ __forceinline__ float sigmoidRef(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x))); }

// ref.ops.situ, one IEEE operation a statement as torch runs it: gate = even index, up = odd index of gu.
// gu bf16 [pairs, 2 * inter], gamma float [pairs], y bf16 [pairs, inter].
extern "C" __global__ void tf_ink2l_situ(const __nv_bfloat16* gu, const float* gamma, __nv_bfloat16* y, int pairs, int inter,
                                         float gate_cap, float up_cap) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)pairs * inter) return;
    const size_t p = i / inter;
    const float g = __bfloat162float(gu[2 * i]);
    const float u = __bfloat162float(gu[2 * i + 1]);
    float act = __fmul_rn(__fmul_rn(gate_cap, tanhf(__fdiv_rn(g, gate_cap))), sigmoidRef(g));
    act = __fmul_rn(act, __fmul_rn(up_cap, tanhf(__fdiv_rn(u, up_cap))));
    y[i] = __float2bfloat16_rn(__fmul_rn(act, gamma[p]));
}

// out[row, c] = bf16(dn[row, 0, c] + dn[row, 1, c] + ... + dn[row, 7, c]), added in slot order from zero.
extern "C" __global__ void tf_ink2l_ex_sum(const float* dn, __nv_bfloat16* out, int rows, int n_dim) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * n_dim) return;
    const size_t row = i / n_dim;
    const size_t c = i % n_dim;
    float sum = 0.0f;
    for (int slot = 0; slot < kSlots; ++slot) sum = __fadd_rn(sum, dn[(row * kSlots + slot) * n_dim + c]);
    out[i] = __float2bfloat16_rn(sum);
}
