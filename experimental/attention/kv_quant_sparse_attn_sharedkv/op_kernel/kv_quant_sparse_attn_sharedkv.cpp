/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file kv_quant_sparse_attn_sharedkv.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kv_quant_sparse_attn_sharedkv_template_tiling_key.h"
#include "arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel.h"
#include "arch35/kv_quant_sparse_attn_sharedkv_common_arch35.h"
#include "kv_quant_sparse_attn_sharedkv_common.h"
// #include "kv_quant_sparse_attn_sharedkv_scfa.h"
// #include "sparse_attn_sharedkv_cfa.h"

using namespace AscendC;

#define SAS_OP_IMPL(templateClass, tilingdataClass, ...)                                          \
    do {                                                                                          \
        templateClass<SASType<__VA_ARGS__>> op;                                                   \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                     \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in;                          \
        op.Init(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ,  \
                seqUsedKV, sinks, metadata, attentionOut, user, tiling_data, tiling, &tPipe);    \
        op.Process();                                                                             \
    } while (0)


template<typename Q_T, typename KV_T, typename T, ImplModeEnum implMode, LayOutTypeEnum layout,
    S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
    DTemplateType dVTemplateType, typename OUTPUT_T, bool isPa, bool isFd>
 __global__ __aicore__ void
kv_quant_sparse_attn_sharedkv(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                       __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
                       __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
                       __gm__ uint8_t *seqUsedKV, __gm__ uint8_t *sinks,
                       __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut,
                       __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    SAS_OP_IMPL(BaseApi::KvQuantSparseAttnSharedkvScfa, KvQuantSparseAttnSharedkvTilingData, bfloat16_t, 
            fp8_e4m3fn_t, float, ImplModeEnum::AA_HIGH_PRECISION, static_cast<LayOutTypeEnum>(layout), 
            static_cast<S1TemplateType>(s1TemplateType), static_cast<S2TemplateType>(s2TemplateType),
            static_cast<DTemplateType>(dTemplateType), static_cast<DTemplateType>(dVTemplateType), 
            bfloat16_t, true, false);

    // if constexpr (ORIG_DTYPE_Q == DT_FLOAT16 && (ORIG_DTYPE_ORI_KV == DT_FLOAT16 || ORIG_DTYPE_CMP_KV == DT_FLOAT16) &&
    //               ORIG_DTYPE_ATTN_OUT == DT_FLOAT16) {
    //     SAS_OP_IMPL(SparseAttnSharedkvScfa, SparseAttnSharedkvTilingData, half, half, half,
    //         FLASH_DECODE, static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T));
    // } else { // bf16
    //     SAS_OP_IMPL(SparseAttnSharedkvScfa, SparseAttnSharedkvTilingData, bfloat16_t, bfloat16_t, bfloat16_t,
    //         FLASH_DECODE, static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T));
    // }
}