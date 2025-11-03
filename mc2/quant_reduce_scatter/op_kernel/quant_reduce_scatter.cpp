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
 * \file quant_reduce_scatter.cpp
 * \brief
 */
#include <kernel_operator.h>
#include <lib/matmul_intf.h>
#include "quant_reduce_scatter_tiling.h"
#include "quant_reduce_scatter.h"

using namespace AscendC;
using namespace QuantReduceScatterImpl;

extern "C" __global__ __aicore__ void quant_reduce_scatter(GM_ADDR x, GM_ADDR scales, GM_ADDR xOut, GM_ADDR workspaceGM,
                                                           GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(QuantReduceScatterTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantReduceScatterTilingData, tilingData, tilingGM);
    TPipe pipe;
    QuantReduceScatter<true, true, true> op;
    op.Init(x, scales, xOut, workspaceGM, &pipe, &tilingData);
    op.Process();
}
