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
 * \file moe_token_unpermute_with_routing_map_grad.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "moe_token_unpermute_with_routing_map_grad_prob_not_none_drop_pad_true.h"
#include "moe_token_unpermute_with_routing_map_grad_prob_not_none_drop_pad_false.h"
#include "moe_token_unpermute_with_routing_map_grad_prob_none_drop_pad_true.h"
#include "moe_token_unpermute_with_routing_map_grad_prob_none_drop_pad_false.h"

using namespace MoeTokenUnpermuteWithRoutingMapGrad;

extern "C" __global__ __aicore__ void moe_token_unpermute_with_routing_map_grad(
    GM_ADDR unpermuted_tokens_grad, GM_ADDR out_index, GM_ADDR permute_token_id, GM_ADDR routing_map,
    GM_ADDR permuted_tokens, GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0)) {
        MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadFalse<DTYPE_PERMUTED_TOKENS, int32_t> op;
        op.Init(
            unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs,
            permuted_tokens_grad, probs_grad, tiling_data);
        op.Process();
    }
    if (TILING_KEY_IS(10)) {
        MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadTrue<DTYPE_PERMUTED_TOKENS, int32_t> op;
        op.Init(
            unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs,
            permuted_tokens_grad, probs_grad, tiling_data);
        op.Process();
    }
#ifdef DTYPE_PROBS
    if (TILING_KEY_IS(1)) {
        MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadFalse<DTYPE_PERMUTED_TOKENS, int32_t> op;
        op.Init(
            unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs,
            permuted_tokens_grad, probs_grad, tiling_data);
        op.Process();
    }
    if (TILING_KEY_IS(11)) {
        MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadTrue<DTYPE_PERMUTED_TOKENS, int32_t> op;
        op.Init(
            unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs,
            permuted_tokens_grad, probs_grad, tiling_data);
        op.Process();
    }
#endif
}