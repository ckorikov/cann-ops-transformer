/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul.cpp
 * \brief
 */
// #include "grouped_matmul_kernel.h"


template <int D_T_A, int D_T_B, int D_T_Y, int TRANS_A, int TRANS_B, int GROUP_LIST_TYPE, int IS_STATIC_TILING_API,
          int A8W4_KERNEL_TEMPLATE, int A16W8_KERNEL_TEMPLATE, int AIV_AIC_RATIO>
__global__ __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
                                          GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList,
                                          GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // AscendCUtils::SetOverflow(1);
    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    // GroupedMatmulTilingData tilingData;
    // GetGroupedMatmulTilingData<D_T_A, D_T_B, D_T_Y, TRANS_A, TRANS_B, GROUP_LIST_TYPE, IS_STATIC_TILING_API,
    //                            A8W4_KERNEL_TEMPLATE, A16W8_KERNEL_TEMPLATE, AIV_AIC_RATIO>(tiling, tilingData);
    // GroupedMatmulKernelImpl<D_T_A, D_T_B, D_T_Y, TRANS_A, TRANS_B, GROUP_LIST_TYPE, IS_STATIC_TILING_API,
    //                         A8W4_KERNEL_TEMPLATE, A16W8_KERNEL_TEMPLATE, AIV_AIC_RATIO>(
    //     x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, y, workspace,
    //     tilingData);
}
