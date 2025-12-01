/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "lower_triangular_inverse.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t X_DIM_LIMIT = 5UL;
constexpr size_t Y_DIM_LIMIT = 5UL;

const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
const std::initializer_list<DataType> Y_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

bool CheckInputOutShape(const aclTensor *x)
{
    int64_t pattern_m = x->GetViewShape().GetDim(3);
    int64_t pattern_n = x->GetViewShape().GetDim(4);

    if (!(pattern_m == 32 || pattern_m == 64 || pattern_m == 128 || pattern_m == 256)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's dim is not support, its dim 3 only support 32, 53, 128, 256, but now is %d", pattern_m);
        return false;
    }

    if (pattern_n != pattern_m) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's dim is not support, its dim 4 only support 32, 53, 128, 256, but now is %d", pattern_m);
        return false;
    }

    return true;
}

aclnnStatus CheckParams(const aclTensor *x, const aclTensor *y)
    {
        // 1. 检查参数是否为空指针
        OP_CHECK_NULL(x, return false);
        OP_CHECK_NULL(y, return false);

        // 2. 校验输入、输出参数维度
        OP_CHECK_WRONG_DIMENSION(x, X_DIM_LIMIT, return false);
        OP_CHECK_WRONG_DIMENSION(y, Y_DIM_LIMIT, return false);

        // 3. 校验输入、输出shape参数
        CHECK_RET(CheckInputOutShape(x), ACLNN_ERR_PARAM_NULLPTR);

        // 4. 检查输入的数据类型是否在支持的数据类型范围之内
        OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(y, Y_DTYPE_SUPPORT_LIST, return false);

    
        return ACLNN_SUCCESS;
    }

static aclnnStatus aclnnLowerTriangularInverseGetWorkspaceSizeCommonProcess(const aclTensor *x, const aclTensor *y, aclOpExecutor *executor)
{
    auto ret = CheckParams(x, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    x = l0op::Contiguous(x, executor);
    CHECK_COND(x  != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous x failed.");

    // 调用l0算子ChunkGatedDeltaRuleInverse进行计算
    auto out = l0op::LowerTriangularInverse(x, y, executor);
    CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto ret1 = l0op::ViewCopy(out, y, executor);
    CHECK_RET(ret1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLowerTriangularInverseGetWorkspaceSize(const aclTensor *x, aclTensor *y, 
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnLowerTriangularInverse, DFX_IN(x), DFX_OUT(y));
    auto uniqueExecutor = CREATE_EXECUTOR();
    auto ret = aclnnLowerTriangularInverseGetWorkspaceSizeCommonProcess(x, y, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);


    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLowerTriangularInverse(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLowerTriangularInverse);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}
#ifdef __cplusplus
}
#endif