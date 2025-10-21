/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_swiglu_quant_base_tiling.cpp
 * \brief
 */
#include "grouped_matmul_swiglu_quant_base_tiling.h"
#include "util/math_util.h"
#include "err/ops_err.h"

using namespace matmul_tiling;

namespace optiling {
namespace GroupedMatmulSwigluQuantV2Tiling {
template <typename T>
static inline auto AlignUp(T a, T base) -> T
{
    if (base == 0) {
        return 0;
    }
    return (a + base - 1) / base * base;
}

int64_t GroupedMatmulSwigluQuantV2BaseTiling::CalMaxRowInUbA8W4(const uint64_t ubSize, const uint64_t n)
{
    const uint64_t ALIGNMENT = 8;
    const float WEIGHT_FACTOR = 8.5;
    const uint64_t ALIGNMENT_TERM_FACTOR = 4;
    const uint64_t LINEAR_TERM_FACTOR = 6;
    const uint64_t CONSTANT_TERM = 64;
    const int64_t MIN_ROW_THRESHOLD = 1;

    // 表达式：8.5 * row * n + 4 * alignUp(row, 8) + 6n + 64 <= ubSize

    // 忽略对齐项的初始估计
    int64_t maxRowEstimate = (ubSize - CONSTANT_TERM - LINEAR_TERM_FACTOR * n) / static_cast<int64_t>(WEIGHT_FACTOR * n);

    // 考虑对齐影响
    uint64_t alignedRow = (maxRowEstimate + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    uint64_t totalSize = static_cast<uint64_t>(WEIGHT_FACTOR * maxRowEstimate * n) + ALIGNMENT_TERM_FACTOR * alignedRow +
                         LINEAR_TERM_FACTOR * n + CONSTANT_TERM;

    // 如果超过UB大小，逐步减少row直到满足条件
    while (totalSize > ubSize && maxRowEstimate > 0) {
        maxRowEstimate--;
        alignedRow = (maxRowEstimate + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        totalSize = static_cast<uint64_t>(WEIGHT_FACTOR * maxRowEstimate * n) + ALIGNMENT_TERM_FACTOR * alignedRow + LINEAR_TERM_FACTOR * n +
                    CONSTANT_TERM;
    }

    if (maxRowEstimate < MIN_ROW_THRESHOLD) {
        OP_LOGE(context_->GetNodeName(), "GMM_SWIGLU_QUANT TILING: No valid row found for n = %lu, ubSize = %lu\n", n,
                ubSize);
        return 0;
    }
    return maxRowEstimate;
}

int64_t GroupedMatmulSwigluQuantV2BaseTiling::CalMaxRowInUb(const uint64_t ubSize, const uint64_t n)
{
    uint64_t tmpBufSize = (n / SWIGLU_REDUCE_FACTOR) * FP32_DTYPE_SIZE;
    uint64_t perchannleBufSize = n * FP32_DTYPE_SIZE * DOUBLE_BUFFER;
    uint64_t reduceMaxResBufSize = BLOCK_BYTE;
    uint64_t reduceMaxTmpBufSize = BLOCK_BYTE;
    const uint64_t CONSTANT_TERM = 64;
    int64_t remainUbSize = ubSize - tmpBufSize - perchannleBufSize - reduceMaxResBufSize - reduceMaxTmpBufSize;
    int64_t maxRowInUb =
        remainUbSize / (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) / DOUBLE_BUFFER;
    int64_t curUb = DOUBLE_BUFFER * (maxRowInUb * (INT32_DTYPE_SIZE * n + n / SWIGLU_REDUCE_FACTOR) +
                                     AlignUp(maxRowInUb, FP32_BLOCK_SIZE) * FP32_DTYPE_SIZE);
    if (curUb > remainUbSize) {
        // 64 : make sure ub does not excceed maxUbSize after align up to 8
        maxRowInUb = (remainUbSize - CONSTANT_TERM) /
                     (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) / DOUBLE_BUFFER;
    }
    if (maxRowInUb < 1) {
        // when n > (ubSize - 72) / 19 = 10330, maxRowInUb < 1
        OP_LOGE(context_->GetNodeName(), "GMM_SWIGLU_QUANT TILING: n should not be greater than 10240, now is %lu\n", n);
    }
    return maxRowInUb;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BaseTiling::ParseInputAndAttr()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto weightDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    auto wTensor = context_->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wTensor);
    auto xTensor = context_->GetDynamicInputTensor(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensor);
    auto wScaleTensor = context_->GetDynamicInputTensor(WEIGHT_SCALE_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wScaleTensor);
    auto groupListTensor = context_->GetDynamicInputTensor(GROUPLIST_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, groupListTensor);

    ge::DataType xDType = xDesc->GetDataType();
    ge::DataType weightDType = weightDesc->GetDataType();

    isA8W4MSD_ = (xDType == ge::DataType::DT_INT8 && weightDType == ge::DataType::DT_INT4);
    auto compileInfoPtr = context_->GetCompileInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
        OP_LOGE(context_->GetNodeName(), "CompileInfo is nullptr"), return ge::GRAPH_FAILED);

    m_ = xTensor->GetStorageShape().GetDim(0);
    k_ = xTensor->GetStorageShape().GetDim(1);
    if (wTensor->GetStorageShape().GetDimNum() == ND_WEIGHT_DIM_LIMIT) { // ND
        n_ = wTensor->GetStorageShape().GetDim(DIM_2);
    } else if (wTensor->GetStorageShape().GetDimNum() == NZ_WEIGHT_DIM_LIMIT) { // NZ
        n_ = wTensor->GetStorageShape().GetDim(DIM_1) * wTensor->GetStorageShape().GetDim(DIM_4);
    }

    if (wScaleTensor->GetStorageShape().GetDimNum() == PERCHANNEL_WSCALE_DIM_LIMIT) { // perChannel
        quantGroupNum_ = 1;
    } else if (wScaleTensor->GetStorageShape().GetDimNum() == PERGROUP_WSCALE_DIM_LIMIT) { // perGroup
        quantGroupNum_ = wScaleTensor->GetStorageShape().GetDim(1);
    }

    groupNum_ = groupListTensor->GetStorageShape().GetDim(0);

    if (isA8W4MSD_) {
        maxProcessRowNum_ = CalMaxRowInUbA8W4(compileInfoPtr->ubSize_, n_);
    } else {
        maxProcessRowNum_ = CalMaxRowInUb(compileInfoPtr->ubSize_, n_);
    }

    blockDim_ = compileInfoPtr->aicNum_;
    baseM_ = compileInfoPtr->baseM_;
    baseM_ = compileInfoPtr->baseN_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BaseTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Begin Run GMM Swiglu Tiling .");

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    MatmulApiTiling tiling(ascendcPlatform);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBias(false);
    tiling.SetShape(baseM_, baseN_, k_);
    tiling.SetOrgShape(m_, n_, k_);
    tiling.SetBufferSpace(-1, -1, -1);
    OP_CHECK_IF(
        tiling.GetTiling(tilingData_.mmTilingData) == -1,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "grouped_matmul_swiglu_quant_tiling, get tiling failed"),
        return ge::GRAPH_FAILED);

    usrWorkspaceLimut_ = USER_WORKSPACE_LIMIT;
    mLimit_ = 0;
    if (isA8W4MSD_) {
        mLimit_ = ((usrWorkspaceLimut_ / DOUBLE_WORKSPACE_SPLIT) / (k_ * sizeof(int8_t) + 0x2 * n_ * sizeof(half)));
    } else {
        mLimit_ = ((usrWorkspaceLimut_ / DOUBLE_WORKSPACE_SPLIT) / INT32_DTYPE_SIZE) / n_;
    }

    OP_CHECK_IF(mLimit_ <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "mLimit_ is %ld must over then 0.", mLimit_),
                return ge::GRAPH_FAILED);
    tilingData_.set_mLimit(mLimit_);
    int workSpaceMTemp = (mLimit_ * DOUBLE_WORKSPACE_SPLIT > m_ ? m_ : mLimit_ * DOUBLE_WORKSPACE_SPLIT);
    tilingData_.set_workSpaceOffset1(workSpaceMTemp * k_ * sizeof(int8_t));
    tilingData_.set_workSpaceOffset2(DOUBLE_ROW * workSpaceMTemp * n_ * sizeof(half));
    if (isA8W4MSD_) {
        workspaceSize_ = SYS_WORKSPACE_SIZE +        // 系统预留16MB
            (workSpaceMTemp * k_ * sizeof(int8_t)) + // 第一阶段 预处理左矩阵 (mLimit_, K) * int8 * 2(double WorkSpace)
            (DOUBLE_ROW * workSpaceMTemp * n_ *
             sizeof(half)); // 第二阶段 矩阵乘结果 (2 * mLimit_, N) * fp16 * 2(double WorkSpace)
    } else {
        workspaceSize_ = SYS_WORKSPACE_SIZE + (workSpaceMTemp * n_ * sizeof(int32_t));
    }

    isSplitWorkSpace_ = m_ > mLimit_ * DOUBLE_WORKSPACE_SPLIT;
    SetTilingKeyAndScheMode();
    FillTilingData();
    PrintTilingData();
    OP_LOGD(context_->GetNodeName(), "End Run GMM Swiglu Tiling.");
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV2BaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

void GroupedMatmulSwigluQuantV2BaseTiling::FillTilingData()
{
    tilingData_.set_groupNum(groupNum_);
    tilingData_.set_coreNum(blockDim_);
    tilingData_.set_K(k_);
    tilingData_.set_N(n_);
    tilingData_.set_M(m_);
    tilingData_.set_maxProcessRowNum(maxProcessRowNum_);
    tilingData_.set_groupListLen(groupNum_);
    tilingData_.set_tokenLen(n_);

    tilingData_.set_quantGroupNum(quantGroupNum_);
}

void GroupedMatmulSwigluQuantV2BaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "grouped_matmul_swiglu_quant_tiling.");
    OP_LOGD(context_->GetNodeName(), "groupNum:  %ld",
        tilingData_.get_groupNum());
    OP_LOGD(context_->GetNodeName(), "coreNum:   %u ",
        tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "M:         %ld",
        tilingData_.get_M());
    OP_LOGD(context_->GetNodeName(), "K:         %ld",
        tilingData_.get_K());
    OP_LOGD(context_->GetNodeName(), "N:         %ld",
        tilingData_.get_N());
    OP_LOGD(context_->GetNodeName(), "maxProcessRowNum:    %ld",
        tilingData_.get_maxProcessRowNum());
    OP_LOGD(context_->GetNodeName(), "groupListLen:        %ld",
        tilingData_.get_groupListLen());
    OP_LOGD(context_->GetNodeName(), "tokenLen:            %ld",
        tilingData_.get_tokenLen());
    OP_LOGD(context_->GetNodeName(), "quantGroupNum:       %ld",
        tilingData_.get_quantGroupNum());
    OP_LOGD(context_->GetNodeName(), "USER_WORKSPACE_LIMIT:         %ld", usrWorkspaceLimut_);
    OP_LOGD(context_->GetNodeName(), "mLimit_:                      %ld", mLimit_);
    OP_LOGD(context_->GetNodeName(), "workspaceSizes:               %lu", workspaceSize_);
    OP_LOGD(context_->GetNodeName(), "isSplitWorkSpace:             %s", isSplitWorkSpace_ ? "true" : "false");
}

void GroupedMatmulSwigluQuantV2BaseTiling::SetTilingKeyAndScheMode()
{
    if (isA8W4MSD_) { // A8W4 MSD tiling_key使用4
        tilingKey_ = A8W4_MSD_TILING_KEY_MODE;
        context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    } else if (isSplitWorkSpace_) {
        tilingKey_ = SPLITWORKSPACE_TILING_KEY_MODE;
        context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    } else {
        tilingKey_ = COMMON_TILING_KEY_MODE;
        context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    }
}

ge::graphStatus GroupedMatmulSwigluQuantV2BaseTiling::PostTiling()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(blockDim_);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
        return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

}
}
