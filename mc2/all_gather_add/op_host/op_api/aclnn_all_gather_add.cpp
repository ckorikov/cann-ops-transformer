/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_all_gather_add.h"
#include "securec.h"
#include "acl/acl.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "hccl_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" { 
#endif
static constexpr size_t TWO_DIMS = 2;

extern aclnnStatus aclnnInnerAllGatherAddGetWorkspaceSize(const aclTensor *a, const aclTensor *b, char *group,
                                                          int64_t rankSize, bool isGatherOut, const aclTensor *cOut,
                                                          const aclTensor *gatherOutOut, uint64_t *workspaceSize,
                                                          aclOpExecutor **executor);
extern aclnnStatus aclnnInnerAllGatherAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);
extern "C" void __attribute__((weak))
NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

static aclnnStatus CheckNotNull(const aclTensor *a, const aclTensor *b, 
                         const aclTensor *gatherout, const aclTensor *output)
{
  OP_CHECK_NULL(a, return false);
  OP_CHECK_NULL(b, return false);
  OP_CHECK_NULL(gatherout, return false);
  OP_CHECK_NULL(output, return false);
  return ACLNN_SUCCESS;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16
};

static aclnnStatus CheckDtypeValid(const aclTensor* a, const aclTensor* b, const aclTensor* gatherout, const aclTensor* output)
{
  OP_CHECK_DTYPE_NOT_SUPPORT(a, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(b, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(gatherout, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(output, DTYPE_SUPPORT_LIST, return false);
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const aclTensor *a, const aclTensor *b, const aclTensor *gatherOut, const aclTensor *output)
{
  // TODO：需要检查当前输入的shape是否为本示例shape
    OP_CHECK_WRONG_DIMENSION(a, TWO_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(b, TWO_DIMS, return false);

    if (a->GetViewShape().GetDim(0) != 240 || a->GetViewShape().GetDim(1) != 256
        b->GetViewShape().GetDim(0) != 480 || a->GetViewShape().GetDim(1) != 256) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
        "The current shape is not generalized, which may lead to functional or accuracy issues. Please use the shapes supported by the examples.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto bLen = b->GetViewShape().GetDim(0);
    auto gatherOutLen = gatherOut->GetViewShape().GetDim(0);
    auto outputLen = output->GetViewShape().GetDim(0);

    OP_API_CHECK((bLen != gatherOutLen), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
        "The length of opeator_b and gatherOut should be same, but opeator_b's length is: %ld and gatherOut's length is: %ld.",
        bLen, gatherOutLen);
        return ACLNN_ERR_PARAM_INVALID;
    });

    OP_API_CHECK((gatherOutLen != outputLen), {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
    "The length of output and gatherOut should be same, but outputLen's length is: %ld and gatherOut's length is: %ld.", gatherOutLen, outputLen);
    return ACLNN_ERR_PARAM_INVALID;
    });

    return ACLNN_SUCCESS;
}

static bool IsFormatSupport(const aclTensor* input, Format format, const std::string& inputName)
{
    if (input != nullptr && input->GetStorageFormat() != format) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "%s's format should be ND. actual is [%s].", inputName.c_str(),
            op::ToString(input->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckFormatValid(const aclTensor *a, const aclTensor *b, const aclTensor *gatherOut, const aclTensor *output)
{
    CHECK_RET(IsFormatSupport(a, Format::FORMAT_ND, "a"), false);
    CHECK_RET(IsFormatSupport(b, Format::FORMAT_ND, "b"), false);
    CHECK_RET(IsFormatSupport(gatherOut, Format::FORMAT_ND, "gatherOut"), false);
    CHECK_RET(IsFormatSupport(output, Format::FORMAT_ND, "output"), false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *a, const aclTensor *b, const aclTensor *gatherout, const aclTensor *output)
{
  CHECK_RET(CheckNotNull(a, b, gatherout, output), ACLNN_ERR_PARAM_NULLPTR);

  CHECK_RET(CheckDtypeValid(a, b, gatherout, output), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckShape(a, b, output, gatherout), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckFormatValid(a, b, output, gatherout), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static bool IsGatherOut(const aclTensor *gatherOut) {
  OP_CHECK_NULL(gatherOut, return false);
  if (gatherOut->IsEmpty()) {
    // 检查tensor shape的某一维度是否为0，例如{0，16}
    OP_LOGD("AllGatherAdd, get gatherOut is false.");
    return false;
  }
  return true;
}

/*
    1.创建OpExecutor
    2.入参校验
    3.计算结果拷贝
    4.获取计算过程所需workspace大小
*/
aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor *a, const aclTensor *b, char *group,
                                              int64_t rankSize, const aclTensor *cOut,
                                              const aclTensor *gatherOutOut, uint64_t *workspaceSize,
                                              aclOpExecutor **executor) 
{
  auto retParam = CheckParams(a, b, gatherOutOut, cOut);
  CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

  OP_LOGD("A is %s, B is %s.", a->ToString().GetString(), b->ToString().GetString());
  OP_LOGD("Output is %s, gatherOut is %s.", cOut->ToString().GetString(), gatherOutOut->ToString().GetString());

  bool isGatherOut = IsGatherOut(gatherOutOut);
  aclnnStatus ret = aclnnInnerAllGatherAddGetWorkspaceSize(a, b, group, rankSize, isGatherOut,
                                                           cOut, gatherOutOut, workspaceSize, executor);
  OP_LOGD("AllGatherAdd, aclnnInnerGetWorkspaceSize ret = %d.", ret);

  return ret;
}

aclnnStatus aclnnAllGatherAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                              aclrtStream stream)
{
  if (NnopbaseSetHcclServerType) {
    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
  }
  if (workspace == nullptr || workspaceSize == 0UL) {
    OP_LOGD("Skip the api for empty tensor, workspace size %lu.", workspaceSize);
    return ACLNN_SUCCESS;
  }
  auto ret = aclnnInnerAllGatherAdd(workspace, workspaceSize, executor, stream);
  if (ret != 0) {
    OP_LOGE(ACLNN_ERR_INNER, "This is an error in launch aicore");
    return ACLNN_ERR_INNER;
  }

  return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif