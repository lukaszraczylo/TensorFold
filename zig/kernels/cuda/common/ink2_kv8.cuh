// Inkling 2's 8-bit attention cache rows (TF_INK2_KV8): a 128-wide head vector as int8 codes and one fp32 scale (absmax / 127).
// A global layer's K (or V) cache is [kv_heads][ring][128] int8 codes, then [kv_heads][ring] fp32 scales.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>

namespace ink2kv8 {

constexpr float kCodeMax = 127.0f;

// Codes of 4 values a lane (a warp holds one 128-wide head), packed little-endian; `scale` is the head's, equal in every lane.
__device__ __forceinline__ uint32_t quant4(const float* x, float& scale) {
    float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float inv = amax > 0.0f ? kCodeMax / amax : 0.0f;
    uint32_t packed = 0;
    for (int i = 0; i < 4; ++i) {
        const int q = max(-127, min(127, __float2int_rn(x[i] * inv)));
        packed |= (static_cast<uint32_t>(q) & 0xffu) << (8 * i);
    }
    scale = amax / kCodeMax;
    return packed;
}

__device__ __forceinline__ uint16_t roundBf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

// Eight codes (two words) times `scale`, rounded to bf16: the 16 bytes a bf16 tile holds for them.
__device__ __forceinline__ uint4 dequant8(uint2 q, float scale) {
    const uint32_t w[2] = {q.x, q.y};
    uint32_t o[4];
    for (int i = 0; i < 4; ++i) {
        const uint32_t pair = w[i >> 1] >> ((i & 1) * 16);
        const float a = static_cast<float>(static_cast<int8_t>(pair & 0xffu)) * scale;
        const float b = static_cast<float>(static_cast<int8_t>((pair >> 8) & 0xffu)) * scale;
        o[i] = static_cast<uint32_t>(roundBf(a)) | (static_cast<uint32_t>(roundBf(b)) << 16);
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

// The fp32 scales of a cache whose codes start at `codes`: after kv_heads * ring * 128 bytes.
__device__ __forceinline__ const float* scales(const void* codes, int kv_heads, int ring) {
    return reinterpret_cast<const float*>(static_cast<const int8_t*>(codes) + static_cast<size_t>(kv_heads) * ring * 128);
}

}  // namespace ink2kv8
