#ifndef __SwinTransformerLnQkvQuant_TILING_H__
#define __SwinTransformerLnQkvQuant_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"


#ifdef __NPU_TILING__
inline [aicore] void InitSwinTransformerLnQkvQuantTilingData(const __gm__ uint8_t* tiling, SwinTransformerLnQkvQuantTilingData* const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(SwinTransformerLnQkvQuantTilingData) / 4; i++) *(dst + i) = *(src + i);
}
#else
inline void InitSwinTransformerLnQkvQuantTilingData(uint8_t* tiling, SwinTransformerLnQkvQuantTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(SwinTransformerLnQkvQuantTilingData));
}
#endif


#define GET_TILING_DATA(tiling_data, tiling_arg) \
SwinTransformerLnQkvQuantTilingData tiling_data; \
InitSwinTransformerLnQkvQuantTilingData(tiling_arg, &tiling_data)

#endif