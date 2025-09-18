#ifndef __SwinTransformerLnQkvQuant_TILING_H__
#define __SwinTransformerLnQkvQuant_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct SwinTransformerLnQkvQuantBaseInfo {
    uint32_t bSize= 0;
    uint32_t sSize= 0;
    uint32_t hSize= 0;
    uint32_t headNum= 0;
    uint32_t hWinSize= 0;
    uint32_t wWinSize= 0;
    uint32_t sizePerHead= 0;
    uint32_t patchHeight= 0;
    uint32_t patchWeight= 0;
    uint32_t lnBaseM= 0;
    uint32_t lnBaseK= 0;
    uint32_t lnBufferM= 0;
    uint32_t lnBufferK= 0;
    uint32_t lnMSubLoop= 0;
    uint32_t lnKSubLoop= 0;
    uint32_t loopNum= 0;
    uint32_t loopSum= 0;
    uint32_t singleCoreLnBsSize= 0;
    uint32_t lnBufferNum= 0;
    uint32_t resverd1= 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline [aicore] void InitSwinTransformerLnQkvQuantBaseParams(const __gm__ uint8_t* tiling, SwinTransformerLnQkvQuantBaseInfo* const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(SwinTransformerLnQkvQuantBaseInfo) / 4; i++) *(dst + i) = *(src + i);
}
#else
inline void InitSwinTransformerLnQkvQuantBaseParams(uint8_t* tiling, SwinTransformerLnQkvQuantBaseInfo* const_data)
{
    memcpy(const_data, tiling, sizeof(SwinTransformerLnQkvQuantBaseInfo));
}
#endif

#pragma pack(1)
struct SwinTransformerLnQkvQuantMmInfo {
    uint32_t mmSizeM = 0;
    uint32_t mmSizeK= 0;
    uint32_t mmSizeN= 0;
    uint32_t dimNum= 0;
    uint32_t mDim= 0;
    uint32_t nDim= 0;
    uint32_t shareUbForMm= 0;
    uint32_t mmLoopNum= 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline [aicore] void InitSwinTransformerLnQkvQuantSingleCoreParams(const __gm__ uint8_t* tiling, SwinTransformerLnQkvQuantMmInfo* const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(SwinTransformerLnQkvQuantMmInfo) / 4; i++) *(dst + i) = *(src + i);
}
#else
inline void InitSwinTransformerLnQkvQuantSingleCoreParams(uint8_t* tiling, SwinTransformerLnQkvQuantMmInfo* const_data)
{
    memcpy(const_data, tiling, sizeof(SwinTransformerLnQkvQuantMmInfo));
}
#endif

#pragma pack(1)
struct SwinTransformerLnQkvQuantTilingData {
    uint32_t size = 0;
    uint32_t maxCoreNum = 0;
    uint32_t lnBlockNum = 0;
    uint32_t workSpaceSize = 0;
    uint32_t inputSizeSum = 0;
    uint32_t tmpShareBufferForLn = 0;
    uint32_t tmpBufferForQuant = 0;
    uint32_t weightK = 0;
    uint32_t weightN = 0;
    float epsilon = 0;
    SwinTransformerLnQkvQuantBaseInfo opBaseInfo;
    SwinTransformerLnQkvQuantMmInfo mmInfo;
    LayerNormTiling layernormTilingData;
    TCubeTiling mmTilingParams;
};
#pragma pack()

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