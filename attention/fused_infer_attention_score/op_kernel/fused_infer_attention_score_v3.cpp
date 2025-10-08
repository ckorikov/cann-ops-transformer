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
 * \file fused_infer_attention_score_v3.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "fused_infer_attention_score_tilingkey.h"

#ifdef FIA_ENABLE_MLA
// mla模板使用私有tiling结构，框架编译时根据一组DType预编译获取keylist，根据keylist找到对应的tiling结构
// 在这组DType中，若没有mla模板的key，包含mla模板编译会报错：unknown type name 'FusedInferAttentionScoreTilingData'
#if ((ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16)) || \
    ((ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16))
#include "../../common/op_kernel/arch32/fia_kernel_nonquant_mla.h"
#include "../../common/op_kernel/arch32/fia_kernel_nonquant.h"
#endif
#endif // FIA_ENABLE_MLA

using namespace AscendC;

#define INVOKE_FIA_NO_KFC_MLA_OP_IMPL(templateClass, ...)                                                              \
    do {                                                                                                               \
        using CubeBlockType = FiaBlockCubeNonQuantMla<FIAType<__VA_ARGS__>>;                                              \
        using VecBlockType = FiaBlockVecNonQuantMla<FIAType<__VA_ARGS__>>;                                     \
        using FdBlockType = FiaBlockVecFlashDecode<FIAType<__VA_ARGS__>>;                                                  \
        templateClass<FIAType<__VA_ARGS__>, CubeBlockType, VecBlockType, FdBlockType> op;                              \
        FIA_COPY_TILING_DATA(FusedInferAttentionScoreTilingData, tiling);                                              \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths,                           \
            deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,             \
            blockTable, queryPaddingSize, kvPaddingSize,                                                               \
            keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,                          \
            keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,                                                 \
            queryRope, keyRope, keyRopeAntiquantScale,                                                                 \
            attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                              \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(templateClass, ...)                                                            \
    do {                                                                                                               \
        templateClass<FIAType<__VA_ARGS__>> op;                                                                        \
        FIA_COPY_TILING_DATA(FusedInferAttentionScoreTilingData, tiling);                                              \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths,                           \
            deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,             \
            blockTable, queryPaddingSize, kvPaddingSize,                                                               \
            keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,                          \
            keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,                                                 \
            queryRope, keyRope, keyRopeAntiquantScale,                                                                 \
            attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                              \
        op.Process();                                                                                                  \
    } while (0)

#define FIA_COPY_TILING_DATA(tilingDataStruct, tiling)                                                                 \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);                                             \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;

extern "C" __global__ __aicore__ void fused_infer_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
    __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)

#elif (__CCE_AICORE__ == 200)

#else
    TPipe tPipe;

    /*
    获取Op可用WorkSpace空间
    **/
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16)
    // fp16 7buf_nz
   // fp16 7buf_nz
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    // fp16 7buf_nd
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    // Mla PA bf16 kv_BSH_BSND
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    // Mla NoPA bf16 kv_BNSD
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING);
    // Mla NoPA bf16 kv_BSH_BSND
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING);
    // Mla NoPA bf16 kv_TND
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVTND_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVTND_FLASHDECODING_MLA_TILING);
     // Gqa NoQuant PA
    TILING_KEY_IS(103000000000200000);
    TILING_KEY_IS(103000000000300000);
    TILING_KEY_IS(103000000000600000);
    TILING_KEY_IS(103000000000700000);
    TILING_KEY_IS(103000000010200000);
    TILING_KEY_IS(103000000010300000);
    TILING_KEY_IS(103000000010600000);
    TILING_KEY_IS(103000000010700000);
    TILING_KEY_IS(103000000020200000);
    TILING_KEY_IS(103000000020300000);
    TILING_KEY_IS(103000000020600000);
    TILING_KEY_IS(103000000020700000);
 
    TILING_KEY_IS(103000000000200001);
    TILING_KEY_IS(103000000000300001);
    TILING_KEY_IS(103000000000600001);
    TILING_KEY_IS(103000000000700001);
    TILING_KEY_IS(103000000010200001);
    TILING_KEY_IS(103000000010300001);
    TILING_KEY_IS(103000000010600001);
    TILING_KEY_IS(103000000010700001);
    TILING_KEY_IS(103000000020200001);
    TILING_KEY_IS(103000000020300001);
    TILING_KEY_IS(103000000020600001);
    TILING_KEY_IS(103000000020700001);
 
    TILING_KEY_IS(103000000000200003);
    TILING_KEY_IS(103000000000300003);
    TILING_KEY_IS(103000000000600003);
    TILING_KEY_IS(103000000000700003);
    TILING_KEY_IS(103000000010200003);
    TILING_KEY_IS(103000000010300003);
    TILING_KEY_IS(103000000010600003);
    TILING_KEY_IS(103000000010700003);
    TILING_KEY_IS(103000000020200003);
    TILING_KEY_IS(103000000020300003);
    TILING_KEY_IS(103000000020600003);
    TILING_KEY_IS(103000000020700003);

    TILING_KEY_IS(103000000000200005);
    TILING_KEY_IS(103000000000300005);
    TILING_KEY_IS(103000000000600005);
    TILING_KEY_IS(103000000000700005);
    TILING_KEY_IS(103000000010200005);
    TILING_KEY_IS(103000000010300005);
    TILING_KEY_IS(103000000010600005);
    TILING_KEY_IS(103000000010700005);
    TILING_KEY_IS(103000000020200005);
    TILING_KEY_IS(103000000020300005);
    TILING_KEY_IS(103000000020600005);
    TILING_KEY_IS(103000000020700005);

    // Gqa NoQuant Non PA Non Perf
    TILING_KEY_IS(103000000000000000);
    TILING_KEY_IS(103000000010000001);
    TILING_KEY_IS(103000000030000003);
    TILING_KEY_IS(103000000050000005);
    TILING_KEY_IS(103000000000100000);
    TILING_KEY_IS(103000000010100001);
    TILING_KEY_IS(103000000030100003);
    TILING_KEY_IS(103000000050100005);
    TILING_KEY_IS(103000000000400000);
    TILING_KEY_IS(103000000010400001);
    TILING_KEY_IS(103000000030400003);
    TILING_KEY_IS(103000000050400005);
    TILING_KEY_IS(103000000000500000);
    TILING_KEY_IS(103000000010500001);
    TILING_KEY_IS(103000000030500003);
    TILING_KEY_IS(103000000050500005);

    // Gqa NoQuant PA Perf
    TILING_KEY_IS(103000000100200000);
    TILING_KEY_IS(103000000100300000);
    TILING_KEY_IS(103000000100600000);
    TILING_KEY_IS(103000000100700000);
    TILING_KEY_IS(103000000110200000);
    TILING_KEY_IS(103000000110300000);
    TILING_KEY_IS(103000000110600000);
    TILING_KEY_IS(103000000110700000);
    TILING_KEY_IS(103000000120200000);
    TILING_KEY_IS(103000000120300000);
    TILING_KEY_IS(103000000120600000);
    TILING_KEY_IS(103000000120700000);
    TILING_KEY_IS(103000000100200001);
    TILING_KEY_IS(103000000100300001);
    TILING_KEY_IS(103000000100600001);
    TILING_KEY_IS(103000000100700001);
    TILING_KEY_IS(103000000110200001);
    TILING_KEY_IS(103000000110300001);
    TILING_KEY_IS(103000000110600001);
    TILING_KEY_IS(103000000110700001);
    TILING_KEY_IS(103000000120200001);
    TILING_KEY_IS(103000000120300001);
    TILING_KEY_IS(103000000120600001);
    TILING_KEY_IS(103000000120700001);
	
    // Gqa NoQuant Non PA Perf
    TILING_KEY_IS(103000000100000000);
    TILING_KEY_IS(103000000110000001);
    TILING_KEY_IS(103000000100100000);
    TILING_KEY_IS(103000000110100001);
    TILING_KEY_IS(103000000100400000);
    TILING_KEY_IS(103000000110400001);
    TILING_KEY_IS(103000000100500000);
    TILING_KEY_IS(103000000110500001);
// Mla PA fp16 kv_NZ
#if TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING   // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
// Mla PA fp16 kv_BNSD
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
// Mla PA fp16 kv_BSH_BSND
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
// Mla NoPA fp16 kv_BNSD
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
// Mla NoPA fp16 kv_BSH_BSND
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
// Mla NoPA fp16 kv_TND
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVTND_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, false, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVTND_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half,
                                  half, false, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND);
// Gqa NoQuant PA Non Perf
#elif TILING_KEY_VAR == 103000000000200000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                   FIA_LAYOUT::BNSD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000300000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000600000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000700000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010200000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010300000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010600000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010700000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020200000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020300000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020600000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020700000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000200001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                   FIA_LAYOUT::BNSD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000300001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BNSD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000600001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000700001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010200001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010300001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010600001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010700001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020200001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020300001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020600001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020700001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000200003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::TND, false, false,
                                   FIA_LAYOUT::BNSD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000300003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BNSD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000600003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000700003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010200003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010300003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010600003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010700003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020200003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020300003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020600003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020700003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000200005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false,
                                   FIA_LAYOUT::BNSD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000300005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BNSD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000600005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000700005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010200005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010300005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010600005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010700005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020200005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020300005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020600005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000020700005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPrecision);
// Gqa NoQuant Non PA Non Perf
#elif TILING_KEY_VAR == 103000000000000000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010000001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000030000003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::TND, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000050000005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NTD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000100000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010100001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000030100003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::TND, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000050100005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NTD, false, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000400000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010400001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000030400003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::TND, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000050400005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NTD, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000000500000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000010500001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000030500003
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::TND, true, PerformanceMode::HighPrecision);
#elif TILING_KEY_VAR == 103000000050500005
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NTD, true, PerformanceMode::HighPrecision);
// Gqa NoQuant PA Perf
#elif TILING_KEY_VAR == 103000000100200000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                   FIA_LAYOUT::BNSD, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000100300000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000100600000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000100700000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110200000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110300000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110600000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110700000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000120200000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000120300000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000120600000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000120700000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000100200001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                   FIA_LAYOUT::BNSD, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000100300001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BNSD, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000100600001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000100700001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110200001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110300001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110600001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110700001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000120200001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000120300001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000120600001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000120700001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ, true, PerformanceMode::HighPerformance);
// Gqa NoQuant Non PA Prof								
#elif TILING_KEY_VAR == 103000000100000000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110000001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000100100000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110100001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, false, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000100400000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110400001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000100500000
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true, PerformanceMode::HighPerformance);
#elif TILING_KEY_VAR == 103000000110500001
	INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, half, half, half, half, false, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true, PerformanceMode::HighPerformance);

#endif
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16)
   // bfl6 7buf_nz
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    // 7buf_nd
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    // Mla PA bf16 kv_BSH_BSND
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    // Mla NoPA bf16 kv_BNSD
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING);
    // Mla NoPA bf16 kv_BSH_BSND
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING);
    // Mla NoPA bf16 kv_TND
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVTND_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVTND_FLASHDECODING_MLA_TILING);

     // Gqa NoQuant PA
    TILING_KEY_IS(103000000000222220);
    TILING_KEY_IS(103000000000322220);
    TILING_KEY_IS(103000000000622220);
    TILING_KEY_IS(103000000000722220);
    TILING_KEY_IS(103000000010222220);
    TILING_KEY_IS(103000000010322220);
    TILING_KEY_IS(103000000010622220);
    TILING_KEY_IS(103000000010722220);
    TILING_KEY_IS(103000000020222220);
    TILING_KEY_IS(103000000020322220);
    TILING_KEY_IS(103000000020622220);
    TILING_KEY_IS(103000000020722220);
 
    TILING_KEY_IS(103000000000222221);
    TILING_KEY_IS(103000000000322221);
    TILING_KEY_IS(103000000000622221);
    TILING_KEY_IS(103000000000722221);
    TILING_KEY_IS(103000000010222221);
    TILING_KEY_IS(103000000010322221);
    TILING_KEY_IS(103000000010622221);
    TILING_KEY_IS(103000000010722221);
    TILING_KEY_IS(103000000020222221);
    TILING_KEY_IS(103000000020322221);
    TILING_KEY_IS(103000000020622221);
    TILING_KEY_IS(103000000020722221);
 
    TILING_KEY_IS(103000000000222223);
    TILING_KEY_IS(103000000000322223);
    TILING_KEY_IS(103000000000622223);
    TILING_KEY_IS(103000000000722223);
    TILING_KEY_IS(103000000010222223);
    TILING_KEY_IS(103000000010322223);
    TILING_KEY_IS(103000000010622223);
    TILING_KEY_IS(103000000010722223);
    TILING_KEY_IS(103000000020222223);
    TILING_KEY_IS(103000000020322223);
    TILING_KEY_IS(103000000020622223);
    TILING_KEY_IS(103000000020722223);

    TILING_KEY_IS(103000000000222225);
    TILING_KEY_IS(103000000000322225);
    TILING_KEY_IS(103000000000622225);
    TILING_KEY_IS(103000000000722225);
    TILING_KEY_IS(103000000010222225);
    TILING_KEY_IS(103000000010322225);
    TILING_KEY_IS(103000000010622225);
    TILING_KEY_IS(103000000010722225);
    TILING_KEY_IS(103000000020222225);
    TILING_KEY_IS(103000000020322225);
    TILING_KEY_IS(103000000020622225);
    TILING_KEY_IS(103000000020722225);


    // Gqa NoQuant Non PA
    TILING_KEY_IS(103000000000022220);
    TILING_KEY_IS(103000000010022221);
    TILING_KEY_IS(103000000030022223);
    TILING_KEY_IS(103000000050022223);
    TILING_KEY_IS(103000000000122220);
    TILING_KEY_IS(103000000010122221);
    TILING_KEY_IS(103000000030122223);
    TILING_KEY_IS(103000000050122223);
    TILING_KEY_IS(103000000000422220);
    TILING_KEY_IS(103000000010422221);
    TILING_KEY_IS(103000000030422223);
    TILING_KEY_IS(103000000050422223);
    TILING_KEY_IS(103000000000522220);
    TILING_KEY_IS(103000000010522221);
    TILING_KEY_IS(103000000030522223);
    TILING_KEY_IS(103000000050522223);
    TILING_KEY_IS(103000000050022225);
    TILING_KEY_IS(103000000050122225);
// Mla PA bf16 kv_NZ
#if TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_MLA_TILING   // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
// Mla PA bf16 kv_BNSD
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
// Mla PA bf16 kv_BSH_BSND
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
// Mla NoPA bf16 kv_BNSD
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, false, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
// Mla NoPA bf16 kv_BSH_BSND
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
// Mla NoPA bf16 kv_TND
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVTND_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, false, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVTND_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND);

// Gqa NoQuant PA
#elif TILING_KEY_VAR == 103000000000222220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false,
                                   FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000322220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000622220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000000722220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010222220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010322220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010622220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000010722220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000020222220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020322220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020622220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 103000000020722220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::NZ, true);
 
#elif TILING_KEY_VAR == 103000000000222221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false,
                                   FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000322221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000622221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000000722221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010222221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010322221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010622221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000010722221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000020222221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020322221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020622221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 103000000020722221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::NZ, true);
 
#elif TILING_KEY_VAR == 103000000000222223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false, false,
                                   FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000322223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000622223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000000722223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010222223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010322223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010622223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000010722223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000020222223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020322223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020622223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 103000000020722223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::NZ, true);
 
#elif TILING_KEY_VAR == 103000000000222225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false, false,
                                   FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000322225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000000622225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000000722225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010222225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010322225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000010622225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000010722225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000020222225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020322225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == 103000000020622225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NZ, true);
#elif TILING_KEY_VAR == 103000000020722225 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, true, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NZ, true);

// Gqa NoQuant Non PA
#elif TILING_KEY_VAR == 103000000000022220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000010022221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000030022223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::TND);
#elif TILING_KEY_VAR == 103000000050022225
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NTD);
#elif TILING_KEY_VAR == 103000000000122220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == 103000000010122221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == 103000000030122223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::TND);
#elif TILING_KEY_VAR == 103000000050122225
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NTD);
#elif TILING_KEY_VAR == 103000000000422220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010422221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000030422223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::TND, true);
#elif TILING_KEY_VAR == 103000000050422225
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NTD, true);
#elif TILING_KEY_VAR == 103000000000522220 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BNSD, false, false,
                                    FIA_LAYOUT::BNSD, true);
#elif TILING_KEY_VAR == 103000000010522221 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::BSH, false, false,
                                    FIA_LAYOUT::BSH, true);
#elif TILING_KEY_VAR == 103000000030522223 
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::TND, false, false,
                                    FIA_LAYOUT::TND, true);
#elif TILING_KEY_VAR == 103000000050522225
    INVOKE_FIA_GQA_NO_QUANT_OP_IMPL(FiaKernelNonQuant, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::NTD, false, false,
                                    FIA_LAYOUT::NTD, true);

#endif
#endif

#endif
}
