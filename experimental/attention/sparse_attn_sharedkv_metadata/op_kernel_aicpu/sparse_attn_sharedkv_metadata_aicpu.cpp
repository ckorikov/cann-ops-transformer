/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file sparse_attn_sharedkv_metadata_aicpu.cpp
 * \brief
 */

#include "sparse_attn_sharedkv_metadata_aicpu.h"
#include "../../sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_metadata.h"
#include "../../common/aicpu/cpu_context_util.h"
#include <cstdio>
#include <math.h>

namespace aicpu {
uint32_t
SparseAttnSharedkvMetadataCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("SparseAttnSharedkvMetadataCpuKernel::Compute");
  bool success = Prepare(ctx) && BalanceSchedule() && GenMetaData();
  return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool SparseAttnSharedkvMetadataCpuKernel::Prepare(
    CpuKernelContext &ctx) {

  // input
  cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
  sequsedKV_ = ctx.Input(static_cast<uint32_t>(ParamId::equsedKV));
  // output
  metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

  bool requiredAttrs = GetAttrValue(ctx, "batch_size", batchSize_) &&
                       GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                       GetAttrValue(ctx, "num_heads_kv", numHeadsKV_) &&
                       GetAttrValue(ctx, "head_dim", headDim_) &&
                       GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && 
                       GetAttrValue(ctx, "aiv_core_num", aivCoreNum_) &&
                       GetAttrValue(ctx, "soc_version", socVersion_);
  if (!requiredAttrs) {
    return false;
  }

  // attributes optional
  GetAttrValueOpt(ctx, "topk", topk_);
  GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);
  GetAttrValueOpt(ctx, "ori_mask_mode", oriMaskMode_);
  GetAttrValueOpt(ctx, "cmp_mask_mode", cmpMaskMode_);
  GetAttrValueOpt(ctx, "ori_win_left", oriWinLeft_);
  GetAttrValueOpt(ctx, "ori_win_right", oriWinRight_);
  GetAttrValueOpt(ctx, "layout_q", layoutQ_);
  GetAttrValueOpt(ctx, "layout_kv", layoutKV_);
  GetAttrValueOpt(ctx, "has_ori_kv", hasOriKV_);
  GetAttrValueOpt(ctx, "has_cmp_kv", hasCmpKv_);

  return ParamsCheck();
}

bool SparseAttnSharedkvMetadataCpuKernel::ParamsCheck() {
#if 0
  if (actSeqLenQ_ != nullptr) {
    auto shape = actSeqLenQ_->GetTensorShape();
    auto data = actSeqLenQ_->GetData();
    auto dtype = actSeqLenQ_->GetDataType();

    KERNEL_CHECK_NULLPTR(shape, false,
                         "shape of actual_seq_lengths_query is null");
    KERNEL_CHECK_NULLPTR(data, false,
                         "data of actual_seq_lengths_query is null");
    KERNEL_CHECK_FALSE((dtype == DataType::DT_INT32), false,
                       "dtype %u of actual_seq_lengths_query is not int32 ", dtype);

    KERNEL_CHECK_FALSE(
        (shape->GetDims() == 1 && shape->GetDimSize(0) == batchSize_), false,
        "shape of actual_seq_lengths_query is not {%u,}", batchSize_);
  }

  if (actSeqLenKV_ != nullptr) {
    auto shape = actSeqLenKV_->GetTensorShape();
    auto data = actSeqLenKV_->GetData();
    auto dtype = actSeqLenKV_->GetDataType();

    KERNEL_CHECK_NULLPTR(shape, false,
                         "shape of actual_seq_lengths_kv is null");
    KERNEL_CHECK_NULLPTR(data, false, "data of actual_seq_lengths_kv is null");
    KERNEL_CHECK_FALSE((dtype == DataType::DT_INT32), false,
                       "dtype of actual_seq_lengths_kv is not int32");

    KERNEL_CHECK_FALSE(
        (shape->GetDims() == 1 && shape->GetDimSize(0) == batchSize_), false,
        "shape of actual_seq_lengths_query date is not {%u,}", batchSize_);
  }

  if (sparseSeqLenKV_ != nullptr) {
    auto shape = sparseSeqLenKV_->GetTensorShape();
    auto data = sparseSeqLenKV_->GetData();
    auto dtype = sparseSeqLenKV_->GetDataType();

    KERNEL_CHECK_NULLPTR(shape, false,
                         "shape of sparse_seq_lengths_kv is null");
    KERNEL_CHECK_NULLPTR(data, false, "data of sparse_seq_lengths_kv is null");
    KERNEL_CHECK_FALSE((dtype == DataType::DT_INT32), false,
                       "dtype of sparse_seq_lengths_kv is not int32");
    KERNEL_CHECK_FALSE(
        (shape->GetDims() == 1 && shape->GetDimSize(0) == batchSize_), false,
        "shape of sparse_seq_lengths_kv date is not {%u,}", batchSize_);
  }

  KERNEL_CHECK_FALSE((layoutQuery_ == "BSND" || layoutQuery_ == "TND"), false,
                     "layout_query invalid");

  KERNEL_CHECK_FALSE(
      (layoutKV_ == "BSND" || layoutKV_ == "TND" || layoutKV_ == "PA_BSND" || layoutKV_ == "PA_BNSD"),
      false, "layout_kv invalid");
  KERNEL_CHECK_FALSE((sparseMode_ == 0 || sparseMode_ == 3), false,
                     "sparse_mode invalid");
  KERNEL_CHECK_FALSE(
      (attentionMode_ == 0 || attentionMode_ == 1 || attentionMode_ == 2),
      false, "attention_mode invalid");

  KERNEL_CHECK_NULLPTR(metaData_, false, "metadata is null");
  auto shape = metaData_->GetTensorShape();
  auto data = metaData_->GetData();
  auto dtype = metaData_->GetDataType();

  KERNEL_CHECK_NULLPTR(shape, false, "shape of metadata is null");
  KERNEL_CHECK_NULLPTR(data, false, "data of metadata is null");
  KERNEL_CHECK_FALSE((dtype == DataType::DT_INT32), false,
                     "dtype of metadata is not int32");
  KERNEL_CHECK_FALSE((shape->GetDims() == 1 &&
                      shape->GetDimSize(0) == optiling::SFA_META_SIZE),
                     false, "shape of sparse_seq_lengths_kv date is not {%u,}",
                     optiling::SFA_META_SIZE);

  KERNEL_CHECK_FALSE(
      (aicCoreNum_ != 0 && aivCoreNum_ != 0 && aivCoreNum_ % aicCoreNum_ == 0 &&
       aicCoreNum_ <= optiling::CORE_NUM &&
       aivCoreNum_ <= (2 * optiling::CORE_NUM)),
      false, "core num invalid aic:%u aiv:%u", aicCoreNum_,
      aivCoreNum_);  // more limit check with platform-core
#endif
  return true;
}

bool SparseAttnSharedkvMetadataCpuKernel::BalanceSchedule() { return true; }

bool SparseAttnSharedkvMetadataCpuKernel::GenMetaData() { return true; }
namespace {
  static const char *kernelType = "SparseAttnSharedkvMetadata";
  REGISTER_CPU_KERNEL(kernelType, SparseAttnSharedkvMetadataCpuKernel);
}

}; // namespace aicpu

