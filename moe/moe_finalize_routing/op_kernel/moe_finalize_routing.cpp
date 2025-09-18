/**
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

/*!
 * \file moe_finalize_routing.cpp
 * \brief
 */
#include "moe_finalize_routing_fp_db.h"
#include "moe_finalize_routing_bf16.h"
#include "moe_finalize_routing_fp_cuth.h"
#include "moe_finalize_routing_fp_cuth_k2.h"
#include "moe_finalize_routing_fp_cuth_k4.h"
#include "moe_finalize_routing_fp_cutk.h"
#include "moe_finalize_routing_bf16_cutk.h"
#include "moe_finalize_routing_bf16_cuth.h"
#include "moe_finalize_routing_bf16_cuth_k2.h"
#include "moe_finalize_routing_bf16_cuth_k4.h"
#include "moe_finalize_routing_fp_db_all_bias.h"
#include "moe_finalize_routing_bf16_all_bias.h"
#include "moe_finalize_routing_fp_cuth_k2_optimized.h"

using namespace MoeFinalizeRouting;

extern "C" __global__ __aicore__ void moe_finalize_routing(GM_ADDR expandedPermutedRows, GM_ADDR skip1, GM_ADDR skip2,
                                                           GM_ADDR bias, GM_ADDR scales, GM_ADDR expandedSrcToDstRow,
                                                           GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(20000)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCutK<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20001)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCutK<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20002)) {
        MoeFinalizeRouting::MoeFinalizeRoutingBf16CutK<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20003)) {
        MoeFinalizeRouting::MoeFinalizeRoutingBF16<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20004)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpDb<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20005)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpDb<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20006)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuthK2<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20007)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuthK2<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20008)) {
        MoeFinalizeRouting::MoeFinalizeRoutingBf16CuthK2<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20009)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuthK4<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20010)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuthK4<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20011)) {
        MoeFinalizeRouting::MoeFinalizeRoutingBf16CuthK4<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20012)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuth<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20013)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuth<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20014)) {
        MoeFinalizeRouting::MoeFinalizeRoutingBf16Cuth<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20015)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpDbAllBias<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20016)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpDbAllBias<half> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20017)) {
        MoeFinalizeRouting::MoeFinalizeRoutingBF16AllBias<bfloat16_t> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20018)) {
        MoeFinalizeRouting::MoeFinalizeRoutingFpCuthK2Optimized<float> op;
        op.Init(expandedPermutedRows, skip1, skip2, bias, scales, expandedSrcToDstRow, expertForSourceRow, out, userWS,
                &tilingData);
        op.Process();
    }
}
