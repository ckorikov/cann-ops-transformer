/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_quant_reduce_scatter.cpp
 * \brief
 */
#include "aclnn_quant_reduce_scatter.h"
#include "securec.h"
#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "hccl_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum class HcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerQuantReduceScatterGetWorkspaceSize(const aclTensor* x, const aclTensor* scales,
                                                                const char* group, const char* reduceOp,
                                                                uint64_t yDtype, aclTensor* output,
                                                                uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerQuantReduceScatter(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                const aclrtStream stream);
extern "C" void __attribute__((weak)) HcclServerType(void *executor, HcclServerType sType);

namespace {

// 检查入参是否为nullptr
bool CheckNotNull(const aclTensor* x, const aclTensor* scales, const aclTensor* output)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(scales, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}

// 根据API定义，列出T-G量化所能支持的所有dtype
const std::initializer_list<op::DataType> X_DTYPE_TG_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_FLOAT8_E5M2
};
const std::initializer_list<op::DataType> SCALES_DTYPE_TG_SUPPORT_LIST = {
    op::DataType::DT_FLOAT
};

// 根据API定义，列出MX量化所能支持的所有dtype
const std::initializer_list<op::DataType> X_DTYPE_MX_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2
};
const std::initializer_list<op::DataType> SCALES_DTYPE_MX_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E8M0
};

const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT
};

// 检查x、scales、output的数据类型是否在算子的支持列表内
bool CheckTGAllDtypesValid(const aclTensor* x, const aclTensor* scales, const aclTensor* output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_TG_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scales, SCALES_DTYPE_TG_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

bool CheckMXAllDtypesValid(const aclTensor* x, const aclTensor* scales, const aclTensor* output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_MX_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scales, SCALES_DTYPE_MX_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

bool CheckAllDtypesValid(const aclTensor* x, const aclTensor* scales, const aclTensor* output)
{
    return CheckTGAllDtypesValid(x, scales, output) || CheckMXAllDtypesValid(x, scales, output);
}

aclnnStatus CheckParams(const aclTensor* x, const aclTensor* scales, const aclTensor* output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, scales, output), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckAllDtypesValid(x, scales, output), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
}

aclnnStatus aclnnQuantReduceScatterGetWorkspaceSize(const aclTensor* x, const aclTensor* scales, const char* group,
                                                    const char* reduceOp, aclTensor* output, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor)
{
    aclnnStatus retParam = CheckParams(x, scales, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    aclnnStatus ret = aclnnInnerQuantReduceScatterGetWorkspaceSize(x, scales, group, reduceOp, yDtype, output, workspaceSize, executor);
    OP_LOGD("QuantReduceScatter, aclnnnGetWorkspaceSize ret %d.", ret);
    return ret;
}

aclnnStatus aclnnQuantReduceScatter(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    if (HcclServerType) {
        HcclServerType(executor, HcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerQuantReduceScatter(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif