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
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "matmul_util.h"
#include "hccl_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" { 
#endif
static constexpr size_t TWO_DIMS = 2;
typedef struct {
  uint32_t id;
  const char *funcName;
  bool hasReg;
} NnopbaseDfxId;

extern aclnnStatus aclnnInnerAllGatherAddGetWorkspaceSize(const aclTensor *a, const aclTensor *b, char *group,
                                                          int64_t rankSize, bool isGatherOut, const aclTensor *cOut,
                                                          const aclTensor *gatherOutOut, uint64_t *workspaceSize,
                                                          aclOpExecutor **executor);
extern aclnnStatus aclnnInnerAllGatherAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);
extern "C" aclnnStatus NnopbaseGetAttrAddr(void *executor, const size_t index, void **attrAddr, size_t *attrLen);
extern "C" void NnopbaseGetOutputTensorAddr(void *executor, const size_t index, void **addr);
extern "C" void NnopbaseGetInputTensorAddr(void *executor, const size_t index, void **addr);
extern "C" void NnopbaseSetInputTensorAddr(void *executor, const size_t index, const void *const addr);
extern "C" void NnopbaseGetTilingData(void *executor, void **tilingData, uint64_t *dataLen);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void* NnopbaseGetUserHandle(void *executor);
extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern "C" void NnopbaseReportLaunchInfo(const uint64_t beginTime, const char *const opType);
extern "C" aclnnStatus NnopbaseReportAicpuAdditionInfo(const uint64_t timeStamp, const char *const opType);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);
static bool CheckNotNull(const aclTensor *a, const aclTensor *b, 
                         const aclTensor *gatherout, const aclTensor *output)
{
  OP_CHECK_NULL(a, return false);
  OP_CHECK_NULL(b, return false);
  OP_CHECK_NULL(gatherout, return false);
  OP_CHECK_NULL(output, return false);
  return true;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

static bool CheckDtypeValid(const aclTensor* a, const aclTensor* b, const aclTensor* gatherout, const aclTensor* output)
{
  OP_CHECK_DTYPE_NOT_SUPPORT(a, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(b, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(gatherout, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(output, DTYPE_SUPPORT_LIST, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *a, const aclTensor *b, const aclTensor *gatherout, const aclTensor *output)
{
  CHECK_RET(CheckNotNull(a, b, gatherout, output), ACLNN_ERR_PARAM_NULLPTR);

  CHECK_RET(CheckDtypeValid(a, b, gatherout, output), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static bool IsGatherOut(const aclTensor *gatherOut) {
  OP_CHECK_NULL(gatherOut, return false);
  if (gatherOut->IsEmpty()) {//怎么算empty
    OP_LOGD("AllGatherAdd, get gather out is false.");
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor *a, const aclTensor *b, const aclTensor *gatherOut, const aclTensor *output)
{
    OP_CHECK_WRONG_DIMENSION(a, TWO_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(b, TWO_DIMS, return false);

    if (IsGatherOut(gatherOut)) {
    auto bLen = b->GetViewShape().GetDim(0);
    auto gatherOutLen = gatherOut->GetViewShape().GetDim(0);
    OP_API_CHECK((bLen != gatherOutLen), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
        "The length of opeator_b and gatherOut should be same, but opeator_b's length is: %ld and gatherOut's length is: %ld.",
        bLen, gatherOutLen);
        return false;
    });
    }
    auto gatherOutLen = gatherOut->GetViewShape().GetDim(0);
    auto outputLen = output->GetViewShape().GetDim(0);
    OP_API_CHECK((gatherOutLen != outputLen), {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
    "The length of output and gatherOut should be same, but outputLen's length is: %ld and gatherOut's length is: %ld.", gatherOutLen, outputLen);
    return false;
    });

    return true;
}

/*
    1.创建OpExecutor
    2.入参校验
    3.计算结果拷贝
    4.获取计算过程所需workspace大小
*/
aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor *a, const aclTensor *b, char *group,
                                              int64_t rankSize, bool isGatherOut, const aclTensor *cOut,
                                              const aclTensor *gatherOutOut, uint64_t *workspaceSize,
                                              aclOpExecutor **executor) 
{
  uint64_t timeStamp = NnopbaseMsprofSysTime();
  auto retParam = CheckParams(a, b, gatherOutOut, cOut);
  CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

  OP_LOGD("A is %s, B is %s.", a->ToString().GetString(), b->ToString().GetString());
  OP_LOGD("Output is %s, gatherOut is %s.", cOut->ToString().GetString(), gatherOutOut->ToString().GetString());

  int64_t rankSize = 0; // ???
  CHECK_RET(CheckShape(a, b, cOut, gatherOutOut), ACLNN_ERR_PARAM_INVALID);
  bool isGatherOut = IsGatherOut(gatherOutOut);
  aclnnStatus ret = aclnnInnerAllGatherAddGetWorkspaceSize(a, b, group, rankSize, isGatherOut,
                                                           cOut, gatherOutOut, workspaceSize, executor);
  OP_LOGD("AllGatherAdd, aclnnInnerGetWorkspaceSize ret = %d.", ret);
  static NnopbaseDfxId dfxId = {0x60000, __func__, false};
  NnopbaseReportApiInfo(timeStamp, dfxId);
  return ret;
}

aclnnStatus aclnnAllGatherAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                              aclrtStream stream) {
  if (workspace == nullptr || workspaceSize == 0UL) {
    OP_LOGD("Skip the api for empty tensor, workspace size %lu.", workspaceSize);
    return ACLNN_SUCCESS;
  }
  uint64_t timeStamp = NnopbaseMsprofSysTime();
  auto ret = aclnnInnerAllGatherAdd(workspace, workspaceSize, executor, stream);
  if (ret != 0) {
    OP_LOGE(ACLNN_ERR_INNER, "This is an error in launch aicore");
    return ACLNN_ERR_INNER;
  }

  static NnopbaseDfxId dfxId = {0x60000, __func__, false};
  NnopbaseReportApiInfo(timeStamp, dfxId);
  return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif