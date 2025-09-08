/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
#include <algorithm>

#include "aclnn_distribute_barrier.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType : uint32_t {
  NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
  NNOPBASE_HCCL_SERVER_TYPE_MTE,
  NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerDistributeBarrierGetWorkspaceSize(
    const aclTensor* xRef, const char* group, int64_t worldSize,
    uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerDistributeBarrier(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream);
extern "C" void __attribute__((weak))
NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNullStatus(const aclTensor* xRef, const char* group) {
  // 检查必选入参出参为非空
  OP_CHECK_NULL(xRef, return false);
  if (group == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
    return false;
  }

  return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor* xRef, const char* group) {
  CHECK_RET(CheckNullStatus(xRef, group), ACLNN_ERR_PARAM_NULLPTR);
  auto groupStrnLen = strnlen(group, HCCL_GROUP_NAME_MAX);
  if ((groupStrnLen >= HCCL_GROUP_NAME_MAX) || (groupStrnLen == 0)) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
            "Required group name length in range (0, HCCL_GROUP_NAME_MAX), but "
            "it's %zu.",
            strnlen(group, HCCL_GROUP_NAME_MAX));
    return false;
  }

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnDistributeBarrierGetWorkspaceSize(aclTensor* xRef,
                                                   const char* group,
                                                   int64_t worldSize,
                                                   uint64_t* workspaceSize,
                                                   aclOpExecutor** executor)

{
  auto retParam = CheckParams(xRef, group);
  CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
  return aclnnInnerDistributeBarrierGetWorkspaceSize(xRef, group, worldSize,
                                                     workspaceSize, executor);
}

aclnnStatus aclnnDistributeBarrier(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor,
                                   aclrtStream stream) {
  if (NnopbaseSetHcclServerType) {
    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
  }
  return aclnnInnerDistributeBarrier(workspace, workspaceSize, executor,
                                     stream);
}

#ifdef __cplusplus
}
#endif