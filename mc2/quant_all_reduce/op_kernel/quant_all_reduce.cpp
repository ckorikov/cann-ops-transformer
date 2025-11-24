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
 * \file quant_all_reduce.cpp
 * \brief kernel内核实现
 */
#include <kernel_operator.h>
#include <lib/matmul_intf.h>
#include "quant_all_reduce_tiling_data.h"
#include "quant_all_reduce_tiling_key.h"
#include "quant_all_reduce.h"

using namespace AscendC;
using namespace QuantAllReduceImpl;

#if defined(__DAV_C310__)
#endif

template<uint32_t quantAllReduceTemplateId>
__global__ __aicore__ void quant_all_reduce(GM_ADDR x, GM_ADDR scales, GM_ADDR output, GM_ADDR workspaceGM,
                                            GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(QuantAllReduceTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantAllReduceTilingData, tilingData, tilingGM);
    TPipe pipe;
    if (quantAllReduceTemplateId == MTE_COMM) {
        QuantAllReduce<true, true, true> op;
        op.Init(x, scales, output, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}
