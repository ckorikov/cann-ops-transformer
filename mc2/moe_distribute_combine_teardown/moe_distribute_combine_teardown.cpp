/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_distribute_combine_teardown.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_teardown_tiling.h"
#include "moe_distribute_combine_teardown.h"
using namespace AscendC;
using namespace MoeDistributeCombineTeardownImpl;

extern "C" __global__ __aicore__ void moe_distribute_combine_teardown(
    GM_ADDR expandX, GM_ADDR quantExpandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR expertScales,
    GM_ADDR commCmdInfo, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR XOut, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineTeardownTilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(1000)) { // tp=1
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTeardownTilingData, tilingData, tilingGM);
        MoeDistributeCombineTeardown<DTYPE_EXPAND_X, int32_t> op;
        op.Init(
            expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo, xActiveMask, sharedExpertX, XOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif
}