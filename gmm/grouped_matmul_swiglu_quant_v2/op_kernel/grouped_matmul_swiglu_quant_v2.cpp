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
 * \file grouped_matmul_swiglu_quant.cpp
 * \brief
 */

#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "grouped_matmul_swiglu_quant_spilit_fusion.h"
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/grouped_matmul_swiglu_quant_v2_mxquant.h"
#endif
using namespace AscendC;
using namespace matmul;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
using namespace GroupedMatmulDequantSwigluQuant;
#endif

extern "C" __global__ __aicore__ void grouped_matmul_swiglu_quant_v2(GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList,
                                                                  GM_ADDR weight, GM_ADDR weightScale, 
                                                                  GM_ADDR weightAssistanceMatrix,
                                                                  GM_ADDR bias, 
                                                                  GM_ADDR smoothScale, 
                                                                  GM_ADDR y, GM_ADDR yScale,
                                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if (TILING_KEY_IS(20000000000)) { // transX = false, transW = false
        KERNEL_TASK_TYPE(20000000000, KERNEL_TYPE_MIX_AIC_1_2);
        GmmSwigluAswt<Act::Gemm::layout::RowMajor, Act::Gemm::layout::RowMajor>(x, weight, weightScale,
                                                                                xScale, weightAssistanceMatrix,
                                                                                smoothScale, groupList, y,
                                                                                yScale, workspace, tiling);
    } else if (TILING_KEY_IS(20000000001)) { // transX = false, transW = true
        KERNEL_TASK_TYPE(20000000001, KERNEL_TYPE_MIX_AIC_1_2);
        GmmSwigluAswt<Act::Gemm::layout::RowMajor, Act::Gemm::layout::ColumnMajor>(x, weight, weightScale,
                                                                                   xScale, weightAssistanceMatrix,
                                                                                   smoothScale, groupList, y,
                                                                                   yScale, workspace, tiling);
    }
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (TILING_KEY_IS(3)) { // antiquant msd
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(GMMSwigluQuantV2TilingFusionData, tilingData, tiling);
        GET_TILING_DATA_MEMBER(GMMSwigluQuantV2TilingFusionData, matmulTiling, matmulTilingData, tiling);
        GroupedMatmulDequantSwilguQuantFusion op(&tPipe, &tilingData, &matmulTilingData);
        if ASCEND_IS_AIC {
            op.mm.SetSubBlockIdx(0);
            op.mm.Init(&matmulTilingData, &tPipe);
        }

        op.Init(x, weight, weightScale, xScale, weightAssistanceMatrix, groupList, y, yScale, userWorkspace);
        op.Process();
    }
#endif
}
