#ifndef ROPE_MATRIX_TILING_H
#define ROPE_MATRIX_TILING_H
#include <stdint.h>

// define a struct as TCubeTiling, which can be call by both op_device and op_host
struct RopeMatrixTiling
{
    uint32_t blockDim;
    uint32_t b;
    uint32_t n;
    uint32_t s;
    uint32_t d;
};

#endif