/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_entry.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_ENTRY_310_H_
#define FLASH_ATTENTION_SCORE_ENTRY_310_H_
#include "flash_attention_score_drop_mask_adapter_regbase.h"
#include "flash_attention_score_train.h"
#include "flash_attention_score_kernel_train.h"
#include "flash_attention_score_block_cube.h"
#include "flash_attention_score_block_vec_train.h"
#include "matmul_modules/fa_flag_data.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef __DAV_C310_CUBE__ // CUBE 实现

#define REGBASE_COPY_TILING_DATA(tiling)                                                                               \

#define INVOKE_FA_OP_IMPL_BASEAPI(templateClass, vec1ResultSize, qkvSize, ...)                                 \
    do {                                                                                                       \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                    \
        TPipe tPipe;                                                                                           \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockCube<__VA_ARGS__>, BaseApi::FABlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockVecDummy<__VA_ARGS__>, BaseApi::FABlockVecTrain<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op;                                                         \
        op.InitBaseAPI(query, key, value, pse, dropMask, paddingMask, attenMask, prefix, actualSeqLengths,     \
                actualSeqLengthsKv, nullptr, nullptr, nullptr, deqScaleQ, deqScaleK, deqScaleV, nullptr,       \
                nullptr, queryRope,keyRope, softmaxMax, softmaxSum, softmaxOut, nullptr, attentionOut, user,   \
                nullptr, &tPipe);                                                                              \
        op.Process();                                                                                          \
    } while (0)
#else // VECTOR 实现
#define REGBASE_COPY_TILING_DATA(tiling)                                                                       \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling);                \
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = &tilingDataIn;                      \

#define INVOKE_FA_OP_IMPL_BASEAPI(templateClass, vec1ResultSize, qkvSize, ...)                                 \
    do {                                                                                                       \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                    \
        REGBASE_COPY_TILING_DATA(tiling);                                                                      \
        TPipe tPipe;                                                                                           \
        if (tilingData->inputParamsRegbase.needDropMaskOp) {                                                   \
            FlashAttentionScoreDropMaskAdapterRegbase dropMaskAdapter;                                         \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                          \
            dropMaskAdapter.Process();                                                                         \
            tPipe.Reset();                                                                                     \
        }                                                                                                      \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockCube<__VA_ARGS__>, BaseApi::FABlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockVecDummy<__VA_ARGS__>, BaseApi::FABlockVecTrain<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op;                                                         \
        op.InitBaseAPI(query, key, value, pse, dropMask, paddingMask, attenMask, prefix, actualSeqLengths,     \
                actualSeqLengthsKv, nullptr, nullptr, nullptr, deqScaleQ, deqScaleK, deqScaleV, nullptr,       \
                nullptr,queryRope, keyRope, softmaxMax, softmaxSum, softmaxOut, nullptr, attentionOut, user,   \
                tilingData, &tPipe);                                                                           \
        op.Process();                                                                                          \
    } while (0)
#endif

template<uint8_t implMode, uint8_t layout, uint16_t s1TemplateType, uint16_t s2TemplateType,
    uint16_t dTemplateType, uint16_t dvTemplateType, uint8_t pseMode, bool hasAtten, bool hasDrop, bool hasRope,
    uint8_t outDtype, uint8_t regbase>
inline __aicore__ void flash_attention_score_regbase(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx, __gm__ uint8_t *deqScaleQ,
    __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if __CCE_AICORE__ == 310
    if constexpr(dvTemplateType > dTemplateType || (dTemplateType != 192 && hasRope == 1)) {
        return;
    }
    constexpr LayOutTypeEnum layoutTypeEnum = (layout == 0) ? LayOutTypeEnum::None : LayOutTypeEnum::LAYOUT_TND;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    #if (ORIG_DTYPE_QUERY == DT_FLOAT)
        INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, 0, 0, float, float, float, ImplModeEnum(implMode),
            layoutTypeEnum, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
            DTemplateType(dTemplateType), DTemplateType(dvTemplateType), PseTypeEnum(pseMode), hasAtten, hasDrop, hasRope);
        return;
    #endif
    #if (ORIG_DTYPE_QUERY == DT_BF16)
        INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, 0, 0, bfloat16_t, float, bfloat16_t, ImplModeEnum(implMode),
            layoutTypeEnum, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
            DTemplateType(dTemplateType), DTemplateType(dvTemplateType), PseTypeEnum(pseMode), hasAtten, hasDrop, hasRope);
        return;
    #endif
    #if (ORIG_DTYPE_QUERY == DT_FLOAT16)
        INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, 0, 0, half, float, half, ImplModeEnum(implMode),
            layoutTypeEnum, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
            DTemplateType(dTemplateType), DTemplateType(dvTemplateType), PseTypeEnum(pseMode), hasAtten, hasDrop, hasRope);
        return;
    #endif

    // fp8
    #if (ORIG_DTYPE_QUERY == DT_FLOAT8_E5M2)
        constexpr uint64_t vec1ResultSize = s1TemplateType * s2TemplateType;
        constexpr uint64_t qkvSize = MAX(MAX(s1TemplateType, s2TemplateType) * dTemplateType, s2TemplateType * dvTemplateType);
        if constexpr (outDtype == 1) {
            INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, vec1ResultSize, qkvSize, fp8_e5m2_t, float, half, ImplModeEnum(implMode),
                layoutTypeEnum, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
                DTemplateType(dTemplateType), DTemplateType(dvTemplateType), PseTypeEnum(pseMode), hasAtten, hasDrop, hasRope);
            return;
        }
        INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, vec1ResultSize, qkvSize, fp8_e5m2_t, float, bfloat16_t, ImplModeEnum(implMode),
            layoutTypeEnum, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
            DTemplateType(dTemplateType), DTemplateType(dvTemplateType), PseTypeEnum(pseMode), hasAtten, hasDrop, hasRope);
        return;
    #endif
    #if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN)
        constexpr uint64_t vec1ResultSize = s1TemplateType * s2TemplateType;
        constexpr uint64_t qkvSize = MAX(MAX(s1TemplateType, s2TemplateType) * dTemplateType, s2TemplateType * dvTemplateType);
        if constexpr (outDtype == 1) {
            INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, vec1ResultSize, qkvSize, fp8_e4m3fn_t, float, half, ImplModeEnum(implMode),
                layoutTypeEnum, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
                DTemplateType(dTemplateType), DTemplateType(dvTemplateType), PseTypeEnum(pseMode), hasAtten, hasDrop, hasRope);
            return;
        }
        INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, vec1ResultSize, qkvSize, fp8_e4m3fn_t, float, bfloat16_t, ImplModeEnum(implMode),
            layoutTypeEnum, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
            DTemplateType(dTemplateType), DTemplateType(dvTemplateType), PseTypeEnum(pseMode), hasAtten, hasDrop, hasRope);
        return;
    #endif
    #if (ORIG_DTYPE_QUERY == DT_HIFLOAT8)
        constexpr uint64_t vec1ResultSize = s1TemplateType * s2TemplateType;
        constexpr uint64_t qkvSize = MAX(MAX(s1TemplateType, s2TemplateType) * dTemplateType, s2TemplateType * dvTemplateType);
        if constexpr (outDtype == 1) {
            INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, vec1ResultSize, qkvSize, hifloat8_t, float, half, ImplModeEnum(implMode),
                layoutTypeEnum, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
                DTemplateType(dTemplateType), DTemplateType(dvTemplateType), PseTypeEnum(pseMode), hasAtten, hasDrop, hasRope);
            return;
        }
        INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, vec1ResultSize, qkvSize, hifloat8_t, float, bfloat16_t, ImplModeEnum(implMode),
            layoutTypeEnum, S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
            DTemplateType(dTemplateType), DTemplateType(dvTemplateType), PseTypeEnum(pseMode), hasAtten, hasDrop, hasRope);
        return;
    #endif
#endif
}
#endif // end of FLASH_ATTENTION_SCORE_ENTRY_310_H_
