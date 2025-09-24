/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

/* !
 * \file moe_init_routing_v2_grad_apt.cpp
 * \brief
 */
#include "arch35/moe_init_routing_v2_grad_full_load.h"
#include "arch35/moe_init_routing_v2_grad_split_h.h"
#include "arch35/moe_init_routing_v2_grad.h"

using namespace AscendC;
using namespace MoeInitRoutingV2Grad;

#define TILINGKEY_SPLIT_H_DROPLESS 200001
#define TILINGKEY_SPLIT_H_DROP_PAD 200002
#define TILINGKEY_SPLIT_H_ACTIVE 200003
#define TILINGKEY_FULL_LOAD_DROPLESS 300001
#define TILINGKEY_FULL_LOAD_DROP_PAD 300002
#define TILINGKEY_FULL_LOAD_ACTIVE 300003
#define TILINGKEY_REGBASE_DROPLESS 400001
#define TILINGKEY_REGBASE_DROP_PAD 400002
#define TILINGKEY_REGBASE_ACTIVE 400003

extern "C" __global__ __aicore__ void moe_init_routing_v2_grad(
    GM_ADDR gradExpandedX,
    GM_ADDR expandedRowIdx,
    GM_ADDR gradX,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipeOp;
    if (TILING_KEY_IS(TILINGKEY_SPLIT_H_DROPLESS)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseSplitHTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseSplitHTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradSplitHCompute<DTYPE_GRAD_EXPANDED_X> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_SPLIT_H_DROP_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseSplitHTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseSplitHTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradSplitHCompute<DTYPE_GRAD_EXPANDED_X, DROP_PAD_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_SPLIT_H_ACTIVE)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseSplitHTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseSplitHTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradSplitHCompute<DTYPE_GRAD_EXPANDED_X, ACTIVE_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_FULL_LOAD_DROPLESS)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseFullLoadTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseFullLoadTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradFullLoadCompute<DTYPE_GRAD_EXPANDED_X> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_FULL_LOAD_DROP_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseFullLoadTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseFullLoadTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradFullLoadCompute<DTYPE_GRAD_EXPANDED_X, DROP_PAD_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_FULL_LOAD_ACTIVE)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseFullLoadTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseFullLoadTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradFullLoadCompute<DTYPE_GRAD_EXPANDED_X, ACTIVE_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_REGBASE_DROPLESS)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradCompute<DTYPE_GRAD_EXPANDED_X> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_REGBASE_DROP_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradCompute<DTYPE_GRAD_EXPANDED_X, DROP_PAD_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_REGBASE_ACTIVE)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV2GradRegbaseTilingData, tiling_data_in, tiling);
        const MoeInitRoutingV2GradRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        MoeInitRoutingV2GradCompute<DTYPE_GRAD_EXPANDED_X, ACTIVE_MODE> op(tilingData);
        op.Init(gradExpandedX, expandedRowIdx, gradX, &pipeOp);
        op.Process();
    }

    pipeOp.Destroy();
}
