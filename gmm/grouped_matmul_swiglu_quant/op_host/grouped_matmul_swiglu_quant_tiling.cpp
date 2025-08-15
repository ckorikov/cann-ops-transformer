/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_tiling.cpp
 * \brief
 */
#include <climits>
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"
#include "tiling/tiling_base.h"
#include "grouped_matmul_swiglu_quant_tiling.h"
using namespace ge;
using namespace AscendC;
using namespace GroupedMatmulSwigluQuantTiling;

namespace {
  template <typename T>
  static inline auto AlignUp(T a, T base) -> T 
  {
    if (base == 0) {
      return 0;
    }
    return (a + base - 1) / base * base;
  }
} // namespace

namespace optiling {

struct GMMSwigluCompileInfo {
  uint64_t ubSize_ = 0;
  uint32_t aicNum_ = 0;
  uint32_t baseM_ = 128;
  uint32_t baseN_ = 256;
};

static int64_t CalMaxRowInUb(gert::TilingContext* context, const uint64_t ubSize, const uint64_t n) {
  uint64_t tmpBufSize = (n / SWIGLU_REDUCE_FACTOR) * FP32_DTYPE_SIZE; 
  uint64_t perchannleBufSize = n * FP32_DTYPE_SIZE * DOUBLE_BUFFER; 
  int64_t remainUbSize = ubSize - tmpBufSize - perchannleBufSize; 
  int64_t maxRowInUb = remainUbSize / 
      (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) / DOUBLE_BUFFER;
  int64_t curUb = DOUBLE_BUFFER * (
                   maxRowInUb * (INT32_DTYPE_SIZE * n + n / SWIGLU_REDUCE_FACTOR) + 
                   AlignUp(maxRowInUb, FP32_BLOCK_SIZE) * FP32_DTYPE_SIZE);
  if (curUb > remainUbSize) {
    // 64 : make sure ub does not excceed maxUbSize after align up to 8
    maxRowInUb = (remainUbSize - 64) / 
                 (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) / DOUBLE_BUFFER;
  }
  if (maxRowInUb < 1) {
    // when n > (ubSize - 72) / 19 = 10330, maxRowInUb < 1
    OPS_LOG_E(context->GetNodeName(), "GMM_SWIGLU_QUANT TILING: n should not be greater than 10240, now is %lu\n", n); 
  }
  return maxRowInUb;
}

static void SetTilingKey(gert::TilingContext* context, bool isSplitWorkSpace) {
  if(isSplitWorkSpace){
    context->SetTilingKey(1);
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
  } else {
    context->SetTilingKey(0);
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
  }
}

ASCENDC_EXTERN_C graphStatus TilingGMMSwigluQuant(gert::TilingContext* context) {
  // set info
  OPS_LOG_I(context->GetNodeName(), "Begin Run GMM Swiglu Tiling .");
  
  auto compileInfoPtr = context->GetCompileInfo<GMMSwigluCompileInfo>();
  auto xTensor = context->GetInputTensor(X_INDEX);
  OPS_LOG_E_IF_NULL(context, xTensor, return GRAPH_FAILED);
  const int64_t m = xTensor->GetStorageShape().GetDim(0);
  const int64_t k = xTensor->GetStorageShape().GetDim(1);
  auto wTensor = context->GetInputTensor(WEIGHT_INDEX);
  OPS_LOG_E_IF_NULL(context, wTensor, return GRAPH_FAILED);
  const int64_t n = wTensor->GetStorageShape().GetDim(1) * wTensor->GetStorageShape().GetDim(4);
  auto groupListTensor = context->GetInputTensor(GROUPLIST_INDEX);
  OPS_LOG_E_IF_NULL(context, groupListTensor, return GRAPH_FAILED);
  const int64_t groupNum = groupListTensor->GetStorageShape().GetDim(0);
  GMMSwigluQuantTilingData tilingData;
  const int64_t row = CalMaxRowInUb(context, compileInfoPtr->ubSize_, n);
  tilingData.gmmSwigluBaseParams.set_groupNum(groupNum);
  tilingData.gmmSwigluBaseParams.set_coreNum(compileInfoPtr->aicNum_);
  tilingData.gmmSwigluBaseParams.set_K(k);
  tilingData.gmmSwigluBaseParams.set_N(n);
  tilingData.gmmSwigluBaseParams.set_M(m);
  tilingData.gmmSwiglu.set_maxProcessRowNum(row);
  tilingData.gmmSwiglu.set_groupListLen(groupNum);
  tilingData.gmmSwiglu.set_tokenLen(n);
  
  OPS_LOG_D(context->GetNodeName(),"grouped_matmul_swiglu_quant_tiling.");
  OPS_LOG_D(context->GetNodeName(),"gmmSwigluBaseParams.groupNum:  %ld", groupNum);
  OPS_LOG_D(context->GetNodeName(),"gmmSwigluBaseParams.coreNum:   %u ", compileInfoPtr->aicNum_);
  OPS_LOG_D(context->GetNodeName(),"gmmSwigluBaseParams.M:         %ld", m);
  OPS_LOG_D(context->GetNodeName(),"gmmSwigluBaseParams.K:         %ld", k);
  OPS_LOG_D(context->GetNodeName(),"gmmSwigluBaseParams.N:         %ld", n);
  OPS_LOG_D(context->GetNodeName(),"gmmSwiglu.maxProcessRowNum:    %ld", row);
  OPS_LOG_D(context->GetNodeName(),"gmmSwiglu.groupListLen:        %ld", groupNum);
  OPS_LOG_D(context->GetNodeName(),"gmmSwiglu.tokenLen:            %ld", n);
  
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  using namespace matmul_tiling;
  MatmulApiTiling tiling(ascendcPlatform);
  tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
  tiling.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_INT8);
  tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
  tiling.SetBias(false);
  tiling.SetShape(compileInfoPtr->baseM_, compileInfoPtr->baseN_, k);
  tiling.SetOrgShape(m, n, k);
  tiling.SetBufferSpace(-1, -1, -1);
  OPS_ERR_IF(tiling.GetTiling(tilingData.mmTilingData) == -1,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "grouped_matmul_swiglu_quant_tiling, get tiling failed"),
             return GRAPH_FAILED);
  auto workspaceSizes = context->GetWorkspaceSizes(1);
  int64_t usrWorkspaceLimut = USER_WORKSPACE_LIMIT;
  int64_t mLimit = ((usrWorkspaceLimut / DOUBLE_WORKSPACE_SPLIT) / INT32_DTYPE_SIZE) / n;
  OPS_ERR_IF(mLimit <= 0, 
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),"mLimit is %ld must over then 0.", mLimit),
             return GRAPH_FAILED);
  tilingData.gmmSwigluBaseParams.set_mLimit(mLimit);
  workspaceSizes[0] = SYS_WORKSPACE_SIZE + ((mLimit * DOUBLE_WORKSPACE_SPLIT > m \
                      ? m \
                      : mLimit * DOUBLE_WORKSPACE_SPLIT) * n * sizeof(int32_t));
  bool isSplitWorkSpace = m > mLimit * DOUBLE_WORKSPACE_SPLIT;
  OPS_LOG_D(context->GetNodeName(), "USER_WORKSPACE_LIMIT:         %ld", usrWorkspaceLimut);
  OPS_LOG_D(context->GetNodeName(), "mLimit:                       %ld", mLimit);
  OPS_LOG_D(context->GetNodeName(), "workspaceSizes:               %lu", workspaceSizes[0]);
  OPS_LOG_D(context->GetNodeName(), "isSplitWorkSpace:             %s", isSplitWorkSpace ? "true" : "false");
  SetTilingKey(context, isSplitWorkSpace);
  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->SetBlockDim(compileInfoPtr->aicNum_); // block dim is the number of aicube
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  
  OPS_LOG_D(context->GetNodeName(), "End Run GMM Swiglu Tiling.");
  return GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C graphStatus TilingPrepareForGMMSwigluQuant(gert::TilingParseContext* context) {
  // get info
  fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
  OPS_LOG_E_IF_NULL(context, platformInfoPtr, return GRAPH_FAILED);
  auto compileInfoPtr = context->GetCompiledInfo<GMMSwigluCompileInfo>();
  OPS_LOG_E_IF_NULL(context, compileInfoPtr, return GRAPH_FAILED);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  compileInfoPtr->aicNum_ = ascendcPlatform.GetCoreNumAic();
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize_);
  OPS_LOG_D(context->GetNodeName(), "ubSize is %lu, aicNum is %u.", compileInfoPtr->ubSize_, compileInfoPtr->aicNum_);
  return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulSwigluQuant)
.Tiling(TilingGMMSwigluQuant)
.TilingParse<GMMSwigluCompileInfo>(TilingPrepareForGMMSwigluQuant); 
}  // namespace optiling
