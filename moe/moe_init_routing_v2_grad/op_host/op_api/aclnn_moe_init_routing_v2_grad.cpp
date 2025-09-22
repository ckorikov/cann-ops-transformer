/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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

#include "aclnn_moe_init_routing_v2_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMoeInitRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradExpandedX, const aclTensor* expandedRowIdx, int64_t topK, int64_t dropPadMode,
    int64_t activeNum, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeInitRoutingV2Grad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradExpandedX, const aclTensor* expandedRowIdx, int64_t topK, int64_t dropPadMode,
    int64_t activeNum, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return aclnnInnerMoeInitRoutingV2GradGetWorkspaceSize(
        gradExpandedX, expandedRowIdx, topK, dropPadMode, activeNum, out, workspaceSize, executor);
}

aclnnStatus aclnnMoeInitRoutingV2Grad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerMoeInitRoutingV2Grad(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif