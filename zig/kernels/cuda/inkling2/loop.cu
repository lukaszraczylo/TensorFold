// Inkling 2 Mini's looped-engine kernels; the check tool's smoke kernel for now.
#include <cstdint>

extern "C" __global__ void tf_ink2l_fill(uint32_t* out, int n, uint32_t v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = v;
}
