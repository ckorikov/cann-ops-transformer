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
 * \file distribute_barrier.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "distribute_barrier_tiling.h"
#include "distribute_barrier.h"

using namespace AscendC;
using namespace DistributeBarrierImpl;

extern "C" __global__ __aicore__ void distribute_barrier(GM_ADDR xRef, GM_ADDR xRefOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DistributeBarrierTilingData);
    TPipe pipe;

    if (TILING_KEY_IS(10000)) {
        GET_TILING_DATA_WITH_STRUCT(DistributeBarrierTilingData, tilingData, tilingGM);
        DistributeBarrier<DTYPE_X_REF> op;
        op.Init(workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}