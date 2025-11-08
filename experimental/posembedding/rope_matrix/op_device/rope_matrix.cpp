#include "rope_matrix.h"

void run_rope_matrix_kernel_bf16(
    uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* sin, uint8_t* cos, uint8_t* out, uint8_t* workspace, uint8_t* tiling) {
    rope_matrix_kernel_bf16<<<blockDim, nullptr, stream>>>(x, y, sin, cos, out, workspace, tiling);
}