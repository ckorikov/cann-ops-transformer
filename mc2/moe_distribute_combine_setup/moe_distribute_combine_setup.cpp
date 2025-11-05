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
 * \file moe_distribute_combine_setup.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_setup_tiling.h"
#include "moe_distribute_combine_setup.h"
using namespace AscendC;
using namespace MoeDistributeCombineSetupImpl;
extern "C" __global__ __aicore__ void moe_distribute_combine_setup(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR quantExpandX, GM_ADDR commCmdInfoOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineSetupTilingData);
    auto tiling = (__gm__ MoeDistributeCombineSetupTilingData*)tilingGM;
    __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));
    __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(1000)) { // tp=1
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineSetupTilingData, tilingData, tilingGM);
        MoeDistributeCombineSetup<DTYPE_EXPAND_X, int32_t> op;
        op.Init(
            expandX, expertIds, assistInfoForCombine, quantExpandX, commCmdInfoOut, workspaceGM, &pipe, &tilingData,
            mc2InitTiling, mc2CcTiling);
        op.Process();
    }
#endif
}