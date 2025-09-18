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
 * \file aclnn_moe_token_unpermute_grad.cpp
 * \brief
 */
#include "aclnn_moe_token_unpermute_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMoeFinalizeRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradY, const aclTensor* expandedRowIdx, const aclTensor* expandedXOptional,
    const aclTensor* scalesOptional, const aclTensor* expertIdxOptional, const aclTensor* biasOptional,
    int64_t dropPadMode, int64_t activeNum, int64_t expertNum, int64_t expertCapacity,
    const aclTensor* gradExpandedXOut, const aclTensor* gradScalesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeFinalizeRoutingV2Grad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
extern aclnnStatus aclnnInnerMoeTokenUnpermuteGradGetWorkspaceSize(
    const aclTensor* permuteTokens, const aclTensor* unpermutedTokensGrad, const aclTensor* sortedIndices,
    const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional,
    aclTensor* permutedTokensGradOut, aclTensor* probsGradOut, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeTokenUnpermuteGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(
    const aclTensor* permuteTokens, const aclTensor* unpermutedTokensGrad, const aclTensor* sortedIndices,
    const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional,
    aclTensor* permutedTokensGradOut, aclTensor* probsGradOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    static bool useMoeFinalizeRoutingV2Grad = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95;
    if (!useMoeFinalizeRoutingV2Grad) {
        return aclnnInnerMoeTokenUnpermuteGradGetWorkspaceSize(
            permuteTokens, unpermutedTokensGrad, sortedIndices, probsOptional, paddedMode, restoreShapeOptional,
            permutedTokensGradOut, probsGradOut, workspaceSize, executor);
    }
    CHECK_RET(paddedMode == false, ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_NULL(permuteTokens, return ACLNN_ERR_PARAM_NULLPTR);
    auto permuteTokensShape = permuteTokens->GetViewShape();
    CHECK_RET(permuteTokensShape.GetDimNum() > 0, ACLNN_ERR_PARAM_INVALID);
    int64_t activeNum = permuteTokensShape.GetDim(0);
    aclnnStatus ret = aclnnInnerMoeFinalizeRoutingV2GradGetWorkspaceSize(
        unpermutedTokensGrad, sortedIndices, permuteTokens, probsOptional, nullptr, nullptr, 0, activeNum, 0, 0,
        permutedTokensGradOut, probsGradOut, workspaceSize, executor);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(
            ACLNN_ERR_INNER,
            "aclnnMoeTokenUnpermuteGrad calls aclnnInnerMoeFinalizeRoutingV2Grad, please refer to the document for "
            "parameter correspondence.");
    }
    return ret;
}

aclnnStatus aclnnMoeTokenUnpermuteGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    static bool useMoeFinalizeRoutingV2Grad = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95;
    if (!useMoeFinalizeRoutingV2Grad) {
        return aclnnInnerMoeTokenUnpermuteGrad(workspace, workspaceSize, executor, stream);
    }
    aclnnStatus ret = aclnnInnerMoeFinalizeRoutingV2Grad(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(
            ACLNN_ERR_INNER,
            "aclnnMoeTokenUnpermuteGrad calls aclnnInnerMoeFinalizeRoutingV2Grad, please refer to the document for "
            "parameter correspondence.");
        return ret;
    }
}

#ifdef __cplusplus
}
#endif