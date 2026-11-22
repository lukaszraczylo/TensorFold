// Inkling 2 Mini's looped-engine kernels: the check tool's smoke kernel, the activation quantisers and the expert decode path.
// Accumulation order of an output dot: the grouped (prompt) expert kernels add elements k = l, l + 32, ... of the K axis a lane
// with fmaf; the decode expert kernels add 16-value chunks c = l, l + 32, ... in order. Both then run the xor tree 16, 8, 4, 2, 1.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" __global__ void tf_ink2l_fill(uint32_t* out, int n, uint32_t v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = v;
}

// x[i] = float(bf16(x[i])) in place: a reference bf16 result a later kernel reads as fp32 (the dynamic conv taps).
extern "C" __global__ void tf_ink2l_round_bf16(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
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

// The e2m1 nibble of a grid value: the index into {0, .5, 1, 1.5, 2, 3, 4, 6}, the sign in bit 3.
__device__ __forceinline__ uint32_t e2m1Nibble(float v) {
    const float a = fabsf(v);
    const uint32_t idx = (a >= 0.5f) + (a >= 1.0f) + (a >= 1.5f) + (a >= 2.0f) + (a >= 3.0f) + (a >= 4.0f) + (a >= 6.0f);
    return idx | (signbit(v) ? 8u : 0u);
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
    const __nv_bfloat16* x, float* out, float* values, float* scales, float* amax_out, uint8_t* pcodes, uint8_t* pscales,
    int row, int k) {
    __shared__ float red[kWarps];
    const __nv_bfloat16* xr = x + (size_t)row * k;
    float* outr = out != nullptr ? out + (size_t)row * k : nullptr;
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
        if (outr != nullptr) {
            #pragma unroll
            for (int i = 0; i < kNvBlock; ++i) outr[b * kNvBlock + i] = __fmul_rn(__fmul_rn(w.values[i], w.scale), outer);
        }
        if (pcodes != nullptr) {
            uint32_t lo = 0, hi = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                lo |= e2m1Nibble(w.values[i]) << (4 * i);
                hi |= e2m1Nibble(w.values[8 + i]) << (4 * i);
            }
            *reinterpret_cast<uint2*>(pcodes + (size_t)row * (k / 2) + b * 8) = make_uint2(lo, hi);
            pscales[(size_t)row * (k / kNvBlock) + b] = (uint8_t)__nv_cvt_float_to_fp8(w.scale, __NV_SATFINITE, __NV_E4M3);
        }
        if (values != nullptr) {
            scales[(size_t)row * (k / kNvBlock) + b] = w.scale;
            #pragma unroll
            for (int i = 0; i < kNvBlock; ++i) values[(size_t)row * k + b * kNvBlock + i] = w.values[i];
        }
    }
}

extern "C" __global__ void tf_ink2l_actq_nvfp4(const __nv_bfloat16* x, float* out, int rows, int k) {
    if (blockIdx.x < rows) actqNvfp4Row(x, out, nullptr, nullptr, nullptr, nullptr, nullptr, blockIdx.x, k);
}

// The prompt FP4 mma's input: the row's e2m1 codes packed two a byte (low nibble first) at pcodes [rows, k/2], the e4m3 block
// scales at pscales [rows, k/16] and the row amax [rows]; the same quantisation as tf_ink2l_actq_nvfp4. out may be null.
extern "C" __global__ void tf_ink2l_actq_nvfp4_pack(const __nv_bfloat16* x, float* out, float* amax, uint8_t* pcodes,
                                                    uint8_t* pscales, int rows, int k) {
    if (blockIdx.x < rows) actqNvfp4Row(x, out, nullptr, nullptr, amax, pcodes, pscales, blockIdx.x, k);
}

extern "C" __global__ void tf_ink2l_actq_nvfp4_codes(
    const __nv_bfloat16* x, float* out, float* values, float* scales, float* amax, int rows, int k) {
    if (blockIdx.x < rows) actqNvfp4Row(x, out, values, scales, amax, nullptr, nullptr, blockIdx.x, k);
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
    const __nv_bfloat16* x, float* out, float* values, uint8_t* exps, uint8_t* pcodes, int row, int k) {
    const __nv_bfloat16* xr = x + (size_t)row * k;
    float* outr = out != nullptr ? out + (size_t)row * k : nullptr;
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
            if (outr != nullptr) outr[g * kMxBlock + i] = __fmul_rn(qv, scale);
            if (pcodes != nullptr) pcodes[(size_t)row * k + g * kMxBlock + i] = (uint8_t)__nv_cvt_float_to_fp8(qv, __NV_SATFINITE, __NV_E4M3);
            if (values != nullptr) values[(size_t)row * k + g * kMxBlock + i] = qv;
        }
    }
}

extern "C" __global__ void tf_ink2l_actq_mxfp8(const __nv_bfloat16* x, float* out, int rows, int k) {
    if (blockIdx.x < rows) actqMxfp8Row(x, out, nullptr, nullptr, nullptr, blockIdx.x, k);
}

// The prompt FP8 mma's input: e4m3 codes at pcodes [rows, k] and the UE8M0 biased exponents at exps [rows, k/32]; out may be null.
extern "C" __global__ void tf_ink2l_actq_mxfp8_pack(const __nv_bfloat16* x, float* out, uint8_t* pcodes, uint8_t* exps, int rows, int k) {
    if (blockIdx.x < rows) actqMxfp8Row(x, out, nullptr, exps, pcodes, blockIdx.x, k);
}

extern "C" __global__ void tf_ink2l_actq_mxfp8_codes(
    const __nv_bfloat16* x, float* out, float* values, uint8_t* exps, int rows, int k) {
    if (blockIdx.x < rows) actqMxfp8Row(x, out, values, exps, nullptr, blockIdx.x, k);
}

// ---- Expert decode path: GEMV on the stored weights, softcapped SwiGLU, combine ----
// A stage's device template takes the stored format as a WeightFmt; the extern "C" wrappers are the only per-format code.
constexpr int kExWarps = kThreads / 32;  // output columns a block computes
constexpr int kSlots = 8;
constexpr int kExRows = 2;     // columns a warp computes together, sharing each activation chunk
constexpr int kExChunk = 16;   // values a lane decodes per weight load
constexpr int kExDepth = 5;    // chunk loads a lane keeps in flight per column: ceil(144 chunks / 32 lanes)
constexpr uint32_t kFloatInf = 0x7f800000u;

__constant__ float kE2m1Lut[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float bf16Round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float e4m3ToFloat(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}

// High bytes of the fp16 e2m1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6: a byte-permute table indexed by the low three code bits.
constexpr uint32_t kE2m1HalfLo = 0x3E3C3800u;
constexpr uint32_t kE2m1HalfHi = 0x46444240u;

template <class T>
__device__ __forceinline__ uint32_t bits32(T v) { return *reinterpret_cast<uint32_t*>(&v); }

// bf16(a) in the low half, bf16(b) in the high half.
__device__ __forceinline__ uint32_t bf16Pair(float a, float b) { return bits32(__floats2bfloat162_rn(a, b)); }

// acc + x_lo * lo + x_hi * hi, with lo and hi the two bf16 values of `packed` as float32.
__device__ __forceinline__ float fmaPair(uint32_t packed, float x_lo, float x_hi, float acc) {
    acc = __fmaf_rn(x_lo, __uint_as_float(packed << 16), acc);
    return __fmaf_rn(x_hi, __uint_as_float(packed & 0xFFFF0000u), acc);
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
    // The decode GEMV reads a chunk of 16 values (8 bytes, one block scale) a lane. A code times its scale is exact in fp16, so
    // every weight is the same bf16((code * scale) * outer) as dequant; only the k order of the sum differs.
    using Raw = uint2;
    __device__ static Raw load(const uint8_t* wr, int c) { return __ldg(reinterpret_cast<const uint2*>(wr) + c); }
    __device__ static uint32_t scaleOf(const uint8_t* sr, int c) { return sr[c]; }
    __device__ static float dot(Raw w, uint32_t scale_byte, float outer, const float* xv, float acc) {
        const __half2 sc = __half2half2(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)scale_byte, __NV_E4M3)));
        acc = word(w.x, sc, outer, xv, acc);
        return word(w.y, sc, outer, xv + 8, acc);
    }
    // Nibbles 0..7 of a word are elements 0..7; pair i joins nibbles i and i + 4 so their signs land on bits 15 and 31.
    __device__ static float word(uint32_t w, __half2 sc, float outer, const float* xv, float acc) {
        const uint32_t low = __byte_perm(kE2m1HalfLo, kE2m1HalfHi, w & 0x7777u);
        const uint32_t high = __byte_perm(kE2m1HalfLo, kE2m1HalfHi, (w >> 16) & 0x7777u);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t h = (__byte_perm(low, high, ((4 + i) << 12) | (i << 4)) & 0xFF00FF00u) | ((w << (12 - 4 * i)) & 0x80008000u);
            const float2 f = __half22float2(__hmul2(*reinterpret_cast<__half2*>(&h), sc));
            acc = fmaPair(bf16Pair(__fmul_rn(f.x, outer), __fmul_rn(f.y, outer)), xv[i], xv[i + 4], acc);
        }
        return acc;
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
    // A chunk is 16 e4m3 bytes, half of a 32-value scale block.
    using Raw = uint4;
    __device__ static Raw load(const uint8_t* wr, int c) { return __ldg(reinterpret_cast<const uint4*>(wr) + c); }
    __device__ static uint32_t scaleOf(const uint8_t* sr, int c) { return sr[c >> 1]; }
    __device__ static float dot(Raw w, uint32_t biased, float, const float* xv, float acc) {
        const float exp2 = biased == kUe8m0Nan ? __uint_as_float(kFloatInf) : ue8m0Value((int)biased);
        acc = word(w.x, exp2, xv, acc);
        acc = word(w.y, exp2, xv + 4, acc);
        acc = word(w.z, exp2, xv + 8, acc);
        return word(w.w, exp2, xv + 12, acc);
    }
    __device__ static float word(uint32_t w, float exp2, const float* xv, float acc) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(w >> (16 * i)), __NV_E4M3);
            const float2 f = __half22float2(__half2(raw));
            acc = fmaPair(bf16Pair(__fmul_rn(f.x, exp2), __fmul_rn(f.y, exp2)), xv[2 * i], xv[2 * i + 1], acc);
        }
        return acc;
    }
};

// kExRows adjacent columns n0.. of expert e against the activation row x. A lane takes chunks lane, lane + 32, ... of each
// column, kExDepth loads in flight before any is decoded; the lane sums its chunks in order, then the xor tree. sum is valid
// in every lane. Columns at or past n_dim are read from the last column and not stored by the caller.
template <class Fmt>
__device__ void expertColumns(const float* x, int e, int n0, const uint8_t* w, const uint8_t* scale, const float* amax,
                              const float* denominator, int n_dim, int k_dim, float (&sum)[kExRows]) {
    const int lane = threadIdx.x & 31;
    const int chunks = k_dim / kExChunk;
    const float outer = Fmt::expertOuter(amax, denominator, e);
    const uint8_t* wr[kExRows];
    const uint8_t* sr[kExRows];
    #pragma unroll
    for (int r = 0; r < kExRows; ++r) {
        const size_t row = (size_t)e * n_dim + min(n0 + r, n_dim - 1);
        wr[r] = w + row * (k_dim / Fmt::kPack);
        sr[r] = scale + row * (k_dim / Fmt::kBlock);
        sum[r] = 0.0f;
    }
    for (int base = 0; base < chunks; base += 32 * kExDepth) {
        typename Fmt::Raw wv[kExRows][kExDepth];
        uint32_t sv[kExRows][kExDepth];
        #pragma unroll
        for (int j = 0; j < kExDepth; ++j) {
            const int c = base + lane + 32 * j;
            if (c < chunks) {
                #pragma unroll
                for (int r = 0; r < kExRows; ++r) {
                    wv[r][j] = Fmt::load(wr[r], c);
                    sv[r][j] = Fmt::scaleOf(sr[r], c);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < kExDepth; ++j) {
            const int c = base + lane + 32 * j;
            if (c >= chunks) continue;
            float xv[kExChunk];
            #pragma unroll
            for (int q = 0; q < kExChunk / 4; ++q) {
                const float4 v = __ldg(reinterpret_cast<const float4*>(x) + c * (kExChunk / 4) + q);
                xv[4 * q] = v.x; xv[4 * q + 1] = v.y; xv[4 * q + 2] = v.z; xv[4 * q + 3] = v.w;
            }
            #pragma unroll
            for (int r = 0; r < kExRows; ++r) sum[r] = Fmt::dot(wv[r][j], sv[r][j], outer, xv, sum[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < kExRows; ++r) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum[r] = __fadd_rn(sum[r], __shfl_xor_sync(0xffffffffu, sum[r], o));
    }
}

// Grid: x = (row, slot) pairs, y = tiles of kExWarps * kExRows columns. act float [rows, k]; gu bf16 [rows * 8, n].
template <class Fmt>
__device__ void exGateUp(const float* act, const int* pick, const uint8_t* w, const uint8_t* scale, const float* amax,
                         const float* denominator, __nv_bfloat16* gu, int pairs, int n_dim, int k_dim) {
    const int p = blockIdx.x;
    const int n0 = (blockIdx.y * kExWarps + (threadIdx.x >> 5)) * kExRows;
    if (p >= pairs || n0 >= n_dim) return;
    float sum[kExRows];
    expertColumns<Fmt>(act + (size_t)(p / kSlots) * k_dim, pick[p], n0, w, scale, amax, denominator, n_dim, k_dim, sum);
    if ((threadIdx.x & 31) != 0) return;
    for (int r = 0; r < kExRows && n0 + r < n_dim; ++r) gu[(size_t)p * n_dim + n0 + r] = __float2bfloat16_rn(sum[r]);
}

// act float [rows * 8, k] (the quantised SwiGLU output); dn float [rows * 8, n] holds the bf16-rounded value.
template <class Fmt>
__device__ void exDown(const float* act, const int* pick, const uint8_t* w, const uint8_t* scale, const float* amax,
                       const float* denominator, float* dn, int pairs, int n_dim, int k_dim) {
    const int p = blockIdx.x;
    const int n0 = (blockIdx.y * kExWarps + (threadIdx.x >> 5)) * kExRows;
    if (p >= pairs || n0 >= n_dim) return;
    float sum[kExRows];
    expertColumns<Fmt>(act + (size_t)p * k_dim, pick[p], n0, w, scale, amax, denominator, n_dim, k_dim, sum);
    if ((threadIdx.x & 31) != 0) return;
    for (int r = 0; r < kExRows && n0 + r < n_dim; ++r) dn[(size_t)p * n_dim + n0 + r] = bf16Round(sum[r]);
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

// ---- Grouped expert path (prompt rows): (row, slot) pairs sorted by expert, a block a tile of one expert's pairs ----
// Every pair's dot is lane-strided fmaf over k, then the xor tree; the tile shares each dequantised weight across its pairs.
// The decode kernels above sum in chunk order instead, so the two paths agree to rounding, not bit for bit.
constexpr int kGroupPairs = 16;                        // pairs of one expert a block holds
constexpr int kGroupCols = 64;                         // output columns a block computes
constexpr int kGroupPasses = kGroupCols / kExWarps;    // columns a warp computes in turn

// Counting sort of the (row, slot) pairs by expert, stable in pair order. One block; offsets has experts + 1 entries.
extern "C" __global__ void tf_ink2l_group(const int* pick, int rows, int experts, int top, int* offsets, int* items, int* counts) {
    const int pairs = rows * top;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        int n = 0;
        for (int p = 0; p < pairs; ++p) n += pick[p] == e;
        counts[e] = n;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int e = 0; e < experts; ++e) {
            offsets[e] = sum;
            sum += counts[e];
        }
        offsets[experts] = sum;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        int at = offsets[e];
        for (int p = 0; p < pairs; ++p) {
            if (pick[p] == e) items[at++] = p;
        }
    }
}

// Grid: x = column tiles, y = experts, z = tiles of kGroupPairs pairs (an expert gets at most `rows` pairs).
// Pair p reads act row p / act_div; store(p, n, sum) writes the result of column n.
template <class Fmt, class Store>
__device__ void exGrouped(const float* act, int act_div, const int* offsets, const int* items, const uint8_t* w,
                          const uint8_t* scale, const float* amax, const float* denominator, int n_dim, int k_dim, Store store) {
    const int e = blockIdx.y;
    const int begin = offsets[e];
    const int first = blockIdx.z * kGroupPairs;
    const int count = min(offsets[e + 1] - begin - first, kGroupPairs);
    if (count <= 0) return;
    __shared__ int pair[kGroupPairs];
    if (threadIdx.x < count) pair[threadIdx.x] = items[begin + first + threadIdx.x];
    __syncthreads();
    const float* x[kGroupPairs];
    #pragma unroll
    for (int i = 0; i < kGroupPairs; ++i) x[i] = act + (size_t)(pair[i < count ? i : 0] / act_div) * k_dim;
    const float outer = Fmt::expertOuter(amax, denominator, e);
    const int lane = threadIdx.x & 31;
    for (int j = 0; j < kGroupPasses; ++j) {
        const int n = blockIdx.x * kGroupCols + (threadIdx.x >> 5) + j * kExWarps;
        if (n >= n_dim) break;
        const size_t row = (size_t)e * n_dim + n;
        const uint8_t* wr = w + row * (k_dim / Fmt::kPack);
        const uint8_t* sr = scale + row * (k_dim / Fmt::kBlock);
        float acc[kGroupPairs];
        #pragma unroll
        for (int i = 0; i < kGroupPairs; ++i) acc[i] = 0.0f;
        for (int k = lane; k < k_dim; k += 32) {
            const float wv = Fmt::dequant(wr, sr, k, outer);
            #pragma unroll
            for (int i = 0; i < kGroupPairs; ++i) {
                if (i < count) acc[i] = __fmaf_rn(x[i][k], wv, acc[i]);
            }
        }
        #pragma unroll
        for (int i = 0; i < kGroupPairs; ++i) {
            if (i >= count) continue;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[i] = __fadd_rn(acc[i], __shfl_xor_sync(0xffffffffu, acc[i], o));
            if (lane == 0) store(pair[i], n, acc[i]);
        }
    }
}

template <class Fmt>
__device__ void exgGateUp(const float* act, const int* offsets, const int* items, const uint8_t* w, const uint8_t* scale,
                          const float* amax, const float* denominator, __nv_bfloat16* gu, int n_dim, int k_dim) {
    exGrouped<Fmt>(act, kSlots, offsets, items, w, scale, amax, denominator, n_dim, k_dim,
                   [&](int p, int n, float sum) { gu[(size_t)p * n_dim + n] = __float2bfloat16_rn(sum); });
}

template <class Fmt>
__device__ void exgDown(const float* act, const int* offsets, const int* items, const uint8_t* w, const uint8_t* scale,
                        const float* amax, const float* denominator, float* dn, int n_dim, int k_dim) {
    exGrouped<Fmt>(act, 1, offsets, items, w, scale, amax, denominator, n_dim, k_dim,
                   [&](int p, int n, float sum) { dn[(size_t)p * n_dim + n] = bf16Round(sum); });
}

extern "C" __global__ void tf_ink2l_exg_gu_nvfp4(const float* act, const int* offsets, const int* items, const uint8_t* w,
                                                 const uint8_t* scale, const float* amax, const float* denominator,
                                                 __nv_bfloat16* gu, int n_dim, int k_dim) {
    exgGateUp<Nvfp4Fmt>(act, offsets, items, w, scale, amax, denominator, gu, n_dim, k_dim);
}

extern "C" __global__ void tf_ink2l_exg_gu_mxfp8(const float* act, const int* offsets, const int* items, const uint8_t* w,
                                                 const uint8_t* scale, const float* amax, const float* denominator,
                                                 __nv_bfloat16* gu, int n_dim, int k_dim) {
    exgGateUp<Mxfp8Fmt>(act, offsets, items, w, scale, amax, denominator, gu, n_dim, k_dim);
}

extern "C" __global__ void tf_ink2l_exg_down_nvfp4(const float* act, const int* offsets, const int* items, const uint8_t* w,
                                                   const uint8_t* scale, const float* amax, const float* denominator,
                                                   float* dn, int n_dim, int k_dim) {
    exgDown<Nvfp4Fmt>(act, offsets, items, w, scale, amax, denominator, dn, n_dim, k_dim);
}

extern "C" __global__ void tf_ink2l_exg_down_mxfp8(const float* act, const int* offsets, const int* items, const uint8_t* w,
                                                   const uint8_t* scale, const float* amax, const float* denominator,
                                                   float* dn, int n_dim, int k_dim) {
    exgDown<Mxfp8Fmt>(act, offsets, items, w, scale, amax, denominator, dn, n_dim, k_dim);
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

// ---- Loop kernels: cross-pass messages (ref.loop.messages) and controllers (ref.loop.controller + the sublayer norm) ----
// Every dot and every sum of squares is one warp's lane-strided fmaf (or square-then-add) chain and the xor-shuffle tree.
constexpr int kLoopMaxC = 2304;
constexpr int kLoopMaxRank = 64;
constexpr int kStreams = 4;
constexpr int kMsgSources = 2;
constexpr int kMsgReductions = kMsgSources + kStreams * kMsgSources;
constexpr float kLoopEps = 1e-6f;

__device__ __forceinline__ float warpTree(float acc) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    return acc;
}

// Sum of x[k] * x[k] over k, each square rounded before the add as torch's square then mean does; valid in every lane.
template <class Load>
__device__ __forceinline__ float warpSumSq(Load x, int n) {
    float acc = 0.0f;
    for (int k = threadIdx.x & 31; k < n; k += 32) {
        const float v = x(k);
        acc = __fadd_rn(acc, __fmul_rn(v, v));
    }
    return warpTree(acc);
}

template <class A, class B>
__device__ __forceinline__ float warpDot(A a, B b, int n) {
    float acc = 0.0f;
    for (int k = threadIdx.x & 31; k < n; k += 32) acc = __fmaf_rn(a(k), b(k), acc);
    return warpTree(acc);
}

// torch's mean: the sum times the float 1/n; the caller adds eps and takes rsqrt.
__device__ __forceinline__ float invRms(float sum_sq, int n) {
    return rsqrtf(__fadd_rn(__fmul_rn(sum_sq, __fdiv_rn(1.0f, (float)n)), kLoopEps));
}

// One block a row, rows = grid. b0, b1 bf16 [rows, c] (the previous pass's blocks); qa, qkv float [24, c] raw (the two
// cross-pass query tensors, rows 2m and 2m + 1 for core layer m); out float [4, rows, c], streams attn, k, v, mlp.
// Stream s uses qa[2m], qkv[2m], qkv[2m + 1], qa[2m + 1]; `scale` multiplies each query element once.
extern "C" __global__ void tf_ink2l_msg(const __nv_bfloat16* b0, const __nv_bfloat16* b1, const float* qa, const float* qkv, int m,
                                        float scale, float* out, int rows, int c) {
    __shared__ float red[kMsgReductions];
    const int row = blockIdx.x;
    if (row >= rows || c > kLoopMaxC) return;
    const __nv_bfloat16* src[kMsgSources] = {b0 + (size_t)row * c, b1 + (size_t)row * c};
    const float* qrow[kStreams] = {qa + (size_t)(2 * m) * c, qkv + (size_t)(2 * m) * c, qkv + (size_t)(2 * m + 1) * c,
                                   qa + (size_t)(2 * m + 1) * c};
    for (int j = threadIdx.x >> 5; j < kMsgReductions; j += kWarps) {
        float v;
        if (j < kMsgSources) {
            const __nv_bfloat16* x = src[j];
            v = warpSumSq([&](int k) { return __bfloat162float(x[k]); }, c);
        } else {
            const int s = (j - kMsgSources) / kMsgSources;
            const __nv_bfloat16* x = src[(j - kMsgSources) % kMsgSources];
            const float* q = qrow[s];
            v = warpDot([&](int k) { return __bfloat162float(x[k]); }, [&](int k) { return __fmul_rn(q[k], scale); }, c);
        }
        if ((threadIdx.x & 31) == 0) red[j] = v;
    }
    __syncthreads();
    const float inv0 = invRms(red[0], c);
    const float inv1 = invRms(red[1], c);
    float p0[kStreams], p1[kStreams];
    #pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        const float l0 = __fmul_rn(red[kMsgSources + s * kMsgSources], inv0);
        const float l1 = __fmul_rn(red[kMsgSources + s * kMsgSources + 1], inv1);
        const float mx = fmaxf(l0, l1);
        const float e0 = expf(__fsub_rn(l0, mx));
        const float e1 = expf(__fsub_rn(l1, mx));
        const float total = __fadd_rn(e0, e1);
        p0[s] = __fdiv_rn(e0, total);
        p1[s] = __fdiv_rn(e1, total);
    }
    for (int d = threadIdx.x; d < c; d += kThreads) {
        const float v0 = __fmul_rn(__bfloat162float(src[0][d]), inv0);
        const float v1 = __fmul_rn(__bfloat162float(src[1][d]), inv1);
        #pragma unroll
        for (int s = 0; s < kStreams; ++s) out[((size_t)s * rows + row) * c + d] = __fmaf_rn(p1[s], v1, __fmul_rn(p0[s], v0));
    }
}

// Block (kDownThreads) = kDownRanks warps, each block one row and kDownRanks outputs; blocks = rows * rank / kDownRanks.
// dest, msg float [rows, c]; w_down bf16 [rank, 2c]; h bf16 [rows, rank].
// pair = bf16([dest * rsqrt(mean(dest^2) + eps), msg * rsqrt(mean(msg^2) + eps)]); h = bf16(pair . w_down).
// Each block rebuilds the row's pair (a few KB from L2) so the blocks need no cross-block hand-off.
constexpr int kDownRanks = 2;
constexpr int kDownThreads = kDownRanks * 32;

extern "C" __global__ void tf_ink2l_ctl_down(const float* dest, const float* msg, const __nv_bfloat16* w_down, __nv_bfloat16* h,
                                             int rows, int c, int rank) {
    __shared__ float pair[2 * kLoopMaxC];
    __shared__ float inv[2];
    const int per_row = rank / kDownRanks;
    const int row = blockIdx.x / per_row;
    const int r = (blockIdx.x % per_row) * kDownRanks + (threadIdx.x >> 5);
    if (row >= rows || c > kLoopMaxC) return;
    const float* d = dest + (size_t)row * c;
    const float* g = msg + (size_t)row * c;
    const int warp = threadIdx.x >> 5;
    {
        const float* x = warp == 0 ? d : g;
        const float s = warpSumSq([&](int k) { return x[k]; }, c);
        if ((threadIdx.x & 31) == 0) inv[warp] = invRms(s, c);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < 2 * c; k += kDownThreads) {
        pair[k] = k < c ? bf16Round(__fmul_rn(d[k], inv[0])) : bf16Round(__fmul_rn(g[k - c], inv[1]));
    }
    __syncthreads();
    const __nv_bfloat16* wr = w_down + (size_t)r * 2 * c;
    const float sum = warpDot([&](int k) { return pair[k]; }, [&](int k) { return __bfloat162float(wr[k]); }, 2 * c);
    if ((threadIdx.x & 31) == 0) h[(size_t)row * rank + r] = __float2bfloat16_rn(sum);
}

// Blend, stage 1: one warp a column n, kWarps columns a block, blocks = rows * ceil(c / kWarps).
// h bf16 [rows, rank]; w_gain, w_forget bf16 [c, rank]; z float [rows, c] = (1 - tanh(forget)) * dest + tanh(gain) * msg.
extern "C" __global__ void tf_ink2l_ctl_blend_z(const float* dest, const float* msg, const __nv_bfloat16* h, const __nv_bfloat16* w_gain,
                                                const __nv_bfloat16* w_forget, float* z, int rows, int c, int rank) {
    __shared__ float hr[kLoopMaxRank];
    const int per_row = (c + kWarps - 1) / kWarps;
    const int row = blockIdx.x / per_row;
    const int n = (blockIdx.x % per_row) * kWarps + (threadIdx.x >> 5);
    if (row >= rows || c > kLoopMaxC || rank > kLoopMaxRank) return;
    for (int r = threadIdx.x; r < rank; r += kThreads) hr[r] = __bfloat162float(h[(size_t)row * rank + r]);
    __syncthreads();
    if (n >= c) return;
    const __nv_bfloat16* wg = w_gain + (size_t)n * rank;
    const __nv_bfloat16* wf = w_forget + (size_t)n * rank;
    const float gain = warpDot([&](int k) { return hr[k]; }, [&](int k) { return __bfloat162float(wg[k]); }, rank);
    const float forget = warpDot([&](int k) { return hr[k]; }, [&](int k) { return __bfloat162float(wf[k]); }, rank);
    if ((threadIdx.x & 31) == 0) {
        const size_t i = (size_t)row * c + n;
        z[i] = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, tanhf(forget)), dest[i]), __fmul_rn(tanhf(gain), msg[i]));
    }
}

// Blend, stage 2: blocks of kApplyThreads columns, blocks = rows * ceil(c / kApplyThreads). Warp 0 of every block redoes the
// row's ordered sum of squares of z (one warp's lane-strided chain, as before), then
// out = bf16(z * rsqrt(mean(z^2) + eps) * norm_w). norm_w bf16 [c]; out bf16 [rows, c].
constexpr int kApplyThreads = 128;

extern "C" __global__ void tf_ink2l_ctl_blend_apply(const float* z, const __nv_bfloat16* norm_w, __nv_bfloat16* out, int rows, int c) {
    __shared__ float inv;
    const int per_row = (c + kApplyThreads - 1) / kApplyThreads;
    const int row = blockIdx.x / per_row;
    const int n = (blockIdx.x % per_row) * kApplyThreads + threadIdx.x;
    if (row >= rows) return;
    const float* zr = z + (size_t)row * c;
    if (threadIdx.x < 32) {
        const float s = warpSumSq([&](int k) { return zr[k]; }, c);
        if (threadIdx.x == 0) inv = invRms(s, c);
    }
    __syncthreads();
    if (n < c) out[(size_t)row * c + n] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(zr[n], inv), __bfloat162float(norm_w[n])));
}

// ---- Abliteration (ref.ablit.Ablit.apply): y -= (table[layer]) * Q (Q^T y) on one row a block, rank <= kAblitMaxRank ----
// Warp j computes c_j = sum_i q[i, j] * y[i] (lane-strided fmaf chain, xor-shuffle tree); the update sums j in increasing order.
constexpr int kAblitMaxRank = 8;

extern "C" __global__ void tf_ink2l_ablit(const float* q, const float* table, __nv_bfloat16* y, int hidden, int rank, int layer) {
    __shared__ float c[kAblitMaxRank];
    const float s = table[layer];
    if (s == 0.0f || rank > kAblitMaxRank) return;
    __nv_bfloat16* row = y + (size_t)blockIdx.x * hidden;
    const int warp = threadIdx.x >> 5;
    if (warp < rank) {
        const float dot = warpDot([&](int i) { return __bfloat162float(row[i]); }, [&](int i) { return q[(size_t)i * rank + warp]; }, hidden);
        if ((threadIdx.x & 31) == 0) c[warp] = dot;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hidden; i += kThreads) {
        float acc = 0.0f;
        for (int j = 0; j < rank; ++j) acc = __fadd_rn(acc, __fmul_rn(q[(size_t)i * rank + j], c[j]));
        row[i] = __float2bfloat16_rn(__fsub_rn(__bfloat162float(row[i]), __fmul_rn(s, acc)));
    }
}

// ---- Decode glue: the fp32 addend and the unnormed fp32 residual mix ----
__device__ __forceinline__ float bf16Bits(uint16_t x) { return __uint_as_float(static_cast<uint32_t>(x) << 16); }

// out[i] = bf16(a[i]) + bf16(b[i]) in fp32, no rounding (ref attn_res's `last + addend`); b null: the widened a.
extern "C" __global__ void tf_ink2l_mix_add(float* __restrict__ out, const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = b ? __fadd_rn(bf16Bits(a[i]), bf16Bits(b[i])) : bf16Bits(a[i]);
}

// ops.cu's MixArgs, field for field; `out` here is fp32 [nq, rows, C] and `norm`, `eps` are unused.
constexpr int kMixBlocks = 8;
constexpr int kMixQueries = 3;
struct MixF32Args {
    const uint16_t* blocks[kMixBlocks];
    const void* partial;
    const float* query;
    const uint16_t* norm[kMixQueries];
    float* out;
    int n_blocks, partial_bf16, nq, rows, C;
    float eps;
};

// ref.ops.attn_res with w = None: softmax over [blocks, partial] of the rms-normed values dotted with each query
// (already D^-0.5 scaled by the 1/sqrt(C) here), then the weighted sum of the normed values, fp32 and unnormalised.
extern "C" __global__ void __launch_bounds__(256) tf_ink2l_mix_f32(const MixF32Args a) {
    constexpr int kV = 1 + kMixQueries;
    __shared__ float red[kThreads / 32][kV];
    __shared__ float wts[kMixQueries][kMixBlocks + 1];
    __shared__ float rr[kMixBlocks + 1];
    const size_t row = blockIdx.x;
    const int C = a.C, n = a.n_blocks + 1, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    auto entry = [&](int j, int c) -> float {
        if (j < a.n_blocks) return bf16Bits(a.blocks[j][row * C + c]);
        return a.partial_bf16 ? bf16Bits(static_cast<const uint16_t*>(a.partial)[row * C + c]) : static_cast<const float*>(a.partial)[row * C + c];
    };
    for (int j = 0; j < n; ++j) {
        float v[kV] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c = threadIdx.x; c < C; c += kThreads) {
            const float x = entry(j, c);
            v[0] += x * x;
            for (int q = 0; q < a.nq; ++q) v[1 + q] += x * a.query[(size_t)q * C + c];
        }
        for (int i = 0; i < kV; ++i)
            for (int off = 16; off > 0; off >>= 1) v[i] += __shfl_xor_sync(0xffffffffu, v[i], off);
        if (lane == 0)
            for (int i = 0; i < kV; ++i) red[warp][i] = v[i];
        __syncthreads();
        if (threadIdx.x == 0) {
            float t[kV] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int w = 0; w < kThreads / 32; ++w)
                for (int i = 0; i < kV; ++i) t[i] += red[w][i];
            const float r = rsqrtf(t[0] / C + 1e-6f);
            rr[j] = r;
            for (int q = 0; q < a.nq; ++q) wts[q][j] = t[1 + q] * r * rsqrtf(static_cast<float>(C));
        }
        __syncthreads();
    }
    if (threadIdx.x < a.nq) {
        float* l = wts[threadIdx.x];
        float m = -__uint_as_float(kFloatInf), t = 0.0f;
        for (int j = 0; j < n; ++j) m = fmaxf(m, l[j]);
        for (int j = 0; j < n; ++j) t += expf(l[j] - m);
        for (int j = 0; j < n; ++j) l[j] = expf(l[j] - m) / t;
    }
    __syncthreads();
    for (int q = 0; q < a.nq; ++q)
        for (int c = threadIdx.x; c < C; c += kThreads) {
            float h = 0.0f;
            for (int j = 0; j < n; ++j) h += wts[q][j] * (entry(j, c) * rr[j]);
            a.out[((size_t)q * a.rows + row) * C + c] = h;
        }
}
