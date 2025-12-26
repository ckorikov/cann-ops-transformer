/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file groupedmatmul_kernel.cpp
 * \brief
 */

#include "gmm/grouped_matmul/op_kernel/grouped_matmul_utils.h"
#include "gmm/grouped_matmul/op_kernel/grouped_matmul_tiling_key.h"

// 声明核函数（无模板，通过宏确定类型）
extern "C" __global__ __aicore__ void groupedmatmul_kernel(__gm__ uint8_t *x, __gm__ uint8_t *weight, __gm__ uint8_t *bias,
                                                          __gm__ uint8_t *scale, __gm__ uint8_t *offset,
                                                          __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                                          __gm__ uint8_t *groupList, __gm__ uint8_t *perTokenScale,
                                                          __gm__ uint8_t *y, const GMMTilingData *tilingData)
{
    // utils.h 中的宏会根据编译时注入的 ORIG_DTYPE_X 等宏推导常量
    GroupedMatmulKernelImpl<D_T_A, D_T_B, D_T_Y, TRANS_A, TRANS_B, GROUP_LIST_TYPE, IS_STATIC_TILING_API,
                            A8W4_KERNEL_TEMPLATE, A16W8_KERNEL_TEMPLATE, AIV_AIC_RATIO>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, y, tilingData);
    return;
}