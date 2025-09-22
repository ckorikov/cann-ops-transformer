/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/license/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file aclnn_inplace_matmul_all_reduce_add_rms_norm.cpp
 * \brief
 */

#include "aclnn_inplace_matmul_all_reduce_add_rms_norm.h"
#include "matmul_all_reduce_add_rms_norm/op_host/op_api/aclnn_matmul_all_reduce_add_rms_norm.h"
#include "securec.h"

#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* residual, const aclTensor* gamma,
    double epsilon, const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode,
    const aclTensor* normOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize(
        x1, x2, bias, residual, gamma, epsilon, group, reduceOp, commTurn, streamMode, residual, normOut, workspaceSize,
        executor);
}

aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNorm(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    return aclnnMatmulAllReduceAddRmsNorm(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
