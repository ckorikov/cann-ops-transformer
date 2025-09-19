/**
  * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved. reserved.
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
 * \file moe_token_permute_with_routing_map_grad.cpp
 * \brief
 */

#include "moe_token_unpermute.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_token_permute_with_routing_map_grad.h"

extern "C" __global__ __aicore__ void moe_token_permute_with_routing_map_grad(
    GM_ADDR permutedTokenOutPutGrad, GM_ADDR permutedProbsOutPutGradOptional, GM_ADDR sortedIndices,
    GM_ADDR routingMapOptional, GM_ADDR tokensGradOut, GM_ADDR probsGradOutOptional, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(
        MoeTokenPermuteWithRoutingMapGradTilingData, moeTokenPermuteWithRoutingMapGradUnpermuteTilingData, moe_token_unpermute_tiling_data_in,
        tiling);
    GET_TILING_DATA_MEMBER(
        MoeTokenPermuteWithRoutingMapGradTilingData, moeTokenpermuteWithRoutingMapDropPadTilingData,
        drop_tiling_data_in, tiling);
    // 调用unpermute计算， probs始终false
    if (TILING_KEY_IS(0)) {
        KernelMoeTokenUnpermute<DTYPE_PERMUTED_TOKEN_OUTPUT_GRAD, int32_t, DTYPE_PERMUTED_TOKEN_OUTPUT_GRAD, false> op;
        op.Init(permutedTokenOutPutGrad, sortedIndices, nullptr, tokensGradOut, &moe_token_unpermute_tiling_data_in);
        op.Process();
    } else if (TILING_KEY_IS(1000)) {
        KernelMoeTokenPermuteWithRoutingMap<DTYPE_PERMUTED_TOKEN_OUTPUT_GRAD, int32_t, DTYPE_PERMUTED_TOKEN_OUTPUT_GRAD>
            op;
        op.Init(permutedProbsOutPutGradOptional, sortedIndices, probsGradOutOptional, &drop_tiling_data_in);
        op.Process();
    }
    return;
}
