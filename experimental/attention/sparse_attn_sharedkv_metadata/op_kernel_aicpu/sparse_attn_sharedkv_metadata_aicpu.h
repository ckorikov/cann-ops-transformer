/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file sparse_attn_sharedkv_metadata_aicpu.h
 * \brief
 */

#ifndef SPARSE_ATTN_SHAREDKV_METADATA_AICPU_H
#define SPARSE_ATTN_SHAREDKV_METADATA_AICPU_H

#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include <array>
#include <string>
#include <vector>

namespace aicpu {
class SparseAttnSharedkvMetadataCpuKernel : public CpuKernel {
public:
  SparseAttnSharedkvMetadataCpuKernel() = default;
  ~SparseAttnSharedkvMetadataCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

private:
  bool Prepare(CpuKernelContext &ctx);
  bool ParamsCheck();
  bool BalanceSchedule();
  bool GenMetaData();

private:
  // input
  Tensor *cuSeqlensQ_ = nullptr; // optional
  Tensor *sequsedKV_ = nullptr;  // optional
  // output
  Tensor *metaData_ = nullptr;

  // attribute
  uint32_t batchSize_ = 0;
  uint32_t numHeadsQ_ = 0;
  uint32_t numHeadsKV_ = 0;
  uint32_t headDim_ = 0;

  // optional attr
  uint32_t topk_ = 0;
  uint32_t cmpRatio_ = 0;
  uint32_t oriMaskMode_ = 4;
  uint32_t cmpMaskMode_ = 3;
  int32_t oriWinLeft_ = 128;
  uint32_t oriWinRight_ = 0;
  std::string layoutQ_ = "BSND";
  std::string layoutKV_ = "PA_ND";
  bool hasOriKV_ = true;
  bool hasCmpKv_ = true;
  // attr
  uint32_t aicCoreNum_ = 24U;
  uint32_t aivCoreNum_ = 48U;
  std::string socVersion_ = "ascend910B";

private:
  enum class ParamId : uint32_t {
    // input
    cuSeqlensQ = 0,
    equsedKV = 1,
    // output
    metaData = 0,
  };
};
} // namespace aicpu

#endif