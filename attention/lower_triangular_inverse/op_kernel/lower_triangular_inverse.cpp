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
 * \file lower_triangular_inverse.cpp
 * \brief
 */
#include "lower_triangular_inverse.h"

using namespace AscendC;
using namespace matmul;
using namespace LowerTriangularInverse;

extern "C" __global__ __aicore__ void lower_triangular_inverse(GM_ADDR x, GM_ADDR y, GM_ADDR workspaceGM,
                                                                     GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    InitParams initParams{x, y, user, &pipe, &tilingData};
    if (TILING_KEY_IS(0UL)) {
        MT mm;
        mm.Init(&tilingData.matmulTiling, &pipe);
        LowerTriangularMatrixInversion<float32_t> op(mm);
        op.Init(initParams);
        op.Process();
    }

    // AIC处理多矩阵运算求逆
}
