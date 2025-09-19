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

#include "moe_token_permute_with_routing_map_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
static constexpr int64_t GRAD_Y_SHAPE_WITH_GROUP_IDX = 2;
static constexpr int64_t GRAD_Y_SHAPE_NO_GROUP_IDX = 3;

OP_TYPE_REGISTER(MoeTokenPermuteWithRoutingMapGrad);

const std::array<const aclTensor*, 2> MoeTokenPermuteWithRoutingMapGrad(
    const aclTensor* permutedTokenOutputGrad, const aclTensor* permutedProbsOutputGradOptional,
    const aclTensor* sortedIndices, const aclTensor* routingMapOptional, int64_t numExperts, int64_t tokensNum,
    bool dropAndPad, const aclTensor* tokensGradOut, const aclTensor* probsGradOut, aclOpExecutor* executor)
{
    L0_DFX(
        MoeTokenPermuteWithRoutingMapGrad, permutedTokenOutputGrad, permutedProbsOutputGradOptional, sortedIndices,
        routingMapOptional, numExperts, tokensNum, dropAndPad);

    ADD_TO_LAUNCHER_LIST_AICORE(
        MoeTokenPermuteWithRoutingMapGrad,
        OP_INPUT(permutedTokenOutputGrad, permutedProbsOutputGradOptional, sortedIndices, routingMapOptional),
        OP_OUTPUT(tokensGradOut, probsGradOut), OP_ATTR(numExperts, tokensNum, dropAndPad));

    return std::array<const aclTensor*, 2>{tokensGradOut, probsGradOut};
}
} // namespace l0op