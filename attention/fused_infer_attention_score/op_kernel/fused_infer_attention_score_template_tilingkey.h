/**
 * Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_v3_tiling_key.h
 * \brief
 */

#ifndef TEMPLATE_TILING_KEY_FAG_H_
#define TEMPLATE_TILING_KEY_FAG_H_
#include "ascendc/host_api/tiling/template_argument.h"

// kernel通过宏定义隔离dtype编译tilingkey，降低耗时。tiling侧没有相关宏
#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (-1)
#endif

#define ASCENDC_TPL_1_BW 1
#define ASCENDC_TPL_2_BW 2
#define ASCENDC_TPL_3_BW 3
#define ASCENDC_TPL_4_BW 4
#define ASCENDC_TPL_9_BW 9
#define ASCENDC_TPL_19_BW 19

// 可表示的tilingkey范围为64bit，注意不能超过限制
ASCENDC_TPL_ARGS_DECL(FusedInferAttentionScore, // 算子唯一标识，可以opType保持一致
    // bit 0-3 qType 0:FP16 2:BF16 3:INT8 4:INT4
    ASCENDC_TPL_UINT_DECL(qType, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 0, 2, 3, 4),
    // bit 4-7 kvType 0:FP16 2:BF16 3:INT8 4:INT4
    ASCENDC_TPL_UINT_DECL(kvType, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 0, 2, 3, 4),
    // bit 8-9 outType 0:FP16 2:BF16 3:INT8 4:INT4
    ASCENDC_TPL_UINT_DECL(outType, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST, 0, 2, 3, 4),
    // bit 10-13 qLayout 0:BSH/BSND 1:BNSD 2:NZ 3:TND 4:NBSD 5:NTD
    ASCENDC_TPL_UINT_DECL(qLayout, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5),
    // bit 14-17 kvLayout 0:BSH/BSND 1:BNSD 2:NZ 3:TND 4:NBSD 5:NTD
    ASCENDC_TPL_UINT_DECL(kvLayout, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5),
    // bit 18-21 outLayout 0:BSH/BSND 1:BNSD 2:NZ 3:TND 4:NBSD 5:NTD
    ASCENDC_TPL_UINT_DECL(outLayout, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5),
    // bit 22-25 qQuantMode 0:noquant 1:perChannel 2:perToken 3:perTensorHead 4:perTokenHead 5:perTokenPA 6:perTokenHeadPA
    ASCENDC_TPL_UINT_DECL(qQuantMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6),
    // bit 26-29 kQuantMode 0:noquant 1:perChannel 2:perToken 3:perTensorHead 4:perTokenHead 5:perTokenPA 6:perTokenHeadPA
    ASCENDC_TPL_UINT_DECL(kQuantMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6),
    // bit 30-33 vQuantMode 0:noquant 1:perChannel 2:perToken 3:perTensorHead 4:perTokenHead 5:perTokenPA 6:perTokenHeadPA
    ASCENDC_TPL_UINT_DECL(vQuantMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6),
    // bit 34 isFlashDecode
    ASCENDC_TPL_BOOL_DECL(isFlashDecode, 0, 1),
    // bit 35 isPageAttention
    ASCENDC_TPL_BOOL_DECL(isPageAttention, 0, 1),
    // bit 36 isSharedPrefix
    ASCENDC_TPL_BOOL_DECL(isSharedPrefix, 0, 1),
    // bit 37-54 reserved
    ASCENDC_TPL_UINT_DECL(reserved, ASCENDC_TPL_19_BW, ASCENDC_TPL_UI_LIST, 0),
    // bit 55-63 fiaFlag
    ASCENDC_TPL_UINT_DECL(fiaFlag, ASCENDC_TPL_9_BW, ASCENDC_TPL_UI_LIST, 0, 3),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(qType, ASCENDC_TPL_UI_LIST, 0, 2),
        ASCENDC_TPL_UINT_SEL(kvType, ASCENDC_TPL_UI_LIST, 0, 2),
        ASCENDC_TPL_UINT_SEL(outType, ASCENDC_TPL_UI_LIST, 0, 2),
        ASCENDC_TPL_UINT_SEL(qLayout, ASCENDC_TPL_UI_LIST, 0, 1, 3),
        ASCENDC_TPL_UINT_SEL(kvLayout, ASCENDC_TPL_UI_LIST, 0, 1, 2),
        ASCENDC_TPL_UINT_SEL(outLayout, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(qQuantMode, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(kQuantMode, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(vQuantMode, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_BOOL_SEL(isFlashDecode, 0, 1),
        ASCENDC_TPL_BOOL_SEL(isPageAttention, 1),
        ASCENDC_TPL_BOOL_SEL(isSharedPrefix, 0),
        ASCENDC_TPL_UINT_SEL(reserved, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(fiaFlag, ASCENDC_TPL_UI_LIST, 3),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::FusedInferAttentionScoreTilingData)
    ),
);
#endif