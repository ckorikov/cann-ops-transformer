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
 * \file tiling_sink_kernel.cpp
 * \brief
 */

#include "tiling_sink_kernel.h"
#include "register/device_op_impl_registry.h"
#include "device_op_impl_registry_impl.h"
#include "err/ops_err.h"
#include "status.h"
#include "tiling_aicpu_task.h"
#include "fused_infer_attention_score_tiling.h"

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvLaunch(void *args) {
#ifndef ASCEND_OPTILING_UT
  OP_LOGD("", "begin aicpu tiling task");
  if (args == nullptr) {
    OP_LOGE("", "args is nullptr");
    return aicpu::KERNEL_STATUS_PARAM_INVALID;
  }
  tilingsink::TilingAicpuTask *task = reinterpret_cast<tilingsink::TilingAicpuTask*>(args);
  if (task->opType == nullptr) {
    OP_LOGE("", "opType of task is nullptr");
    return aicpu::KERNEL_STATUS_PARAM_INVALID;
  }
  std::string opType = task->opType;
  optiling::SinkTilingFunc func = 
    optiling::DeviceOpImplRegistry::GetSingleton().GetSinkTilingFunc(opType);
  if (func == nullptr) {
    OP_LOGE(opType.c_str(), "func is nullptr, check if op is registered");
    return aicpu::KERNEL_STATUS_PARAM_INVALID;
  }
  if (!task->tilingContext) {
    OP_LOGE(opType.c_str(), "task tilingContext is nullptr");
    return aicpu::KERNEL_STATUS_PARAM_INVALID;
  }

  if ((func)(task->tilingContext) != ge::GRAPH_SUCCESS) {
    OP_LOGE(opType.c_str(), "aicpu tiling func fail");
    return aicpu::KERNEL_STATUS_INNER_ERROR;
  }
  OP_LOGD(opType.c_str(), "aicpu tiling func success");

  if (reinterpret_cast<uint64_t *>(task->notifyAddr) != nullptr) {
    OP_LOGD(opType.c_str(), "notify value is: %d", *reinterpret_cast<uint64_t *>(task->notifyAddr));
    *reinterpret_cast<uint64_t *>(task->notifyAddr) = 1; // 将此地址置1，super kernel场景同步使用
    #ifdef __aarch64__
      // 插入一句汇编，作用是等待所有存储操作及相关缓存和缓冲区维护操作完成，这里是保证写地址notifyAddr的操作完成
      __asm__ __volatile__("dsb st" : : : "memory");
    #endif
    OP_LOGD(opType.c_str(), "notify value after aicpu tiling is: %d", *reinterpret_cast<uint64_t *>(task->notifyAddr));
  }
  OP_LOGD(opType.c_str(), "end aicpu tiling task");
#endif

  return aicpu::KERNEL_STATUS_OK;
}
}

#ifndef ASCEND_OPTILING_UT
DEVICE_IMPL_OP_OPTILING(FusedInferAttentionScore).Tiling(optiling::DeviceDoOpTilingFusedInferAttentionScore);
DEVICE_IMPL_OP_OPTILING(IncreFlashAttention).Tiling(optiling::DeviceDoOpTilingIncreFlashAttention);
#endif