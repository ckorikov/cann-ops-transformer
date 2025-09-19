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
#ifndef MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_H
#define MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_H

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor*, 2> MoeTokenPermuteWithRoutingMapGrad(
    const aclTensor* permutedTokenOutputGrad, const aclTensor* permutedProbsOutputGradOptional,
    const aclTensor* sortedIndices, const aclTensor* routingMapOptional, int64_t numExperts, int64_t tokensNum,
    bool dropAndPad, const aclTensor* tokensGradOut, const aclTensor* probsGradOut, aclOpExecutor* executor);
} // namespace l0op

#endif // MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_H