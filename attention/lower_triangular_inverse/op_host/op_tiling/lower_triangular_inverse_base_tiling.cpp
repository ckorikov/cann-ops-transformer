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
 * \file chunk_gated_delta_rule_inverse_base_tiling.cc
 * \brief
 */
#include "lower_triangular_inverse_base_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "err/ops_err.h"

namespace optiling {
namespace LowerTriangularInverse {
const uint32_t DIM_0 = 0;
const uint32_t DIM_1 = 1;
const uint32_t DIM_2 = 2;
const uint32_t DIM_3 = 3;
const uint32_t DIM_4 = 4;
constexpr uint32_t CV_PARALL_NUM = 4;
constexpr uint64_t RPC_WORKSIZE = 20;
constexpr uint64_t MB_SIZE = 1024 * 1024;
constexpr uint32_t UBRESTBYTES = 1024 * 100;  // for vector compute

ge::graphStatus LowerTriangularInverseBaseTiling::GetInputShape()
{
    if (context_ == nullptr || context_->GetInputShape(0) == nullptr) {
        OP_LOGE(context_->GetNodeName(), "[GetInputShape] invalid input pointer: context_ or context_->GetInputShape(0)");
        return ge::GRAPH_FAILED;
    }

    // 检查x形状
    int64_t mDims[5];
    auto storageShape = context_->GetInputShape(0)->GetStorageShape();
    size_t dimNum = storageShape.GetDimNum();
    mDims[0] = storageShape[DIM_0];
    mDims[1] = storageShape[DIM_1];
    mDims[2] = storageShape[DIM_2];
    mDims[3] = storageShape[DIM_3];
    mDims[4] = storageShape[DIM_4];
    batch_ = static_cast<uint64_t>(mDims[0] * mDims[1] * mDims[2]);
    m_ = static_cast<uint64_t>(mDims[3]);

    return ge::GRAPH_SUCCESS;
}

void LowerTriangularInverseBaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "blockDim: [%d]", tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "batch: [%u]", tilingData_.get_batch());
    OP_LOGD(context_->GetNodeName(), "m: [%u]", tilingData_.get_m());

}

ge::graphStatus LowerTriangularInverseBaseTiling::ParseInputAndAttr()
{
    uint64_t ubSize, l1Size, l0CSize;

    if (GetInputShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "get input shape failed");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get platform info failed");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
    blockDim_ = ascendcPlatform.GetCoreNumAic();
    
    return ge::GRAPH_SUCCESS;
}

void LowerTriangularInverseBaseTiling::FillTilingData()
{
    // 矩阵计算剩余部分
    tilingData_.matmulTiling.set_dbL0C(1);
    tilingData_.matmulTiling.set_stepKa(1);  // 4: L1中左矩阵单次搬运基于baseK的4倍数据
    tilingData_.matmulTiling.set_stepKb(1);  // 4: L1中右矩阵单次搬运基于baseK的4倍数据
    tilingData_.matmulTiling.set_depthA1(1);  // 8: stepKa的两倍，开启double buffer
    tilingData_.matmulTiling.set_depthB1(1);  // 8: stepKb的两倍，开启double buffer
    tilingData_.matmulTiling.set_stepM(1);
    tilingData_.matmulTiling.set_stepN(1);

    tilingData_.set_coreNum(blockDim_);
    tilingData_.set_batch(batch_);
    tilingData_.set_m(m_);
    tilingData_.set_ubRestBytes(ubRestBytes_);  // 126976: 除分配给TQue外剩余给TBuf的大小为126976
}

ge::graphStatus LowerTriangularInverseBaseTiling::TilingProcess()
{
    // 当前最大为128*128的matmul计算, 预留3倍空间适配最大矩阵
    size_t userWorkspaceSize = CV_PARALL_NUM * 128 * 128 * sizeof(float) * blockDim_;
    size_t systemWorkspaceSize = RPC_WORKSIZE * MB_SIZE; // 20M
    ubRestBytes_ = UBRESTBYTES;

    mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm_.SetBias(false);
    mm_.SetDim(1);
    mm_.SetShape(m_, m_, m_);       
    mm_.SetOrgShape(m_, m_, m_);    // 原始MNK
    mm_.SetFixSplit(16, 16, 16);    // BaseMNK
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {
        OP_LOGE(context_->GetNodeName(), "LowerTriangularInverseBaseTiling Get Tiling Failed!, batch, m: %lu, %lu", batch_, m_);
        return ge::GRAPH_FAILED;
    }
    
    // 后续模板按位处理
    tilingKey_ = 0UL;

    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus LowerTriangularInverseBaseTiling::DoOpTiling()
{
    auto inputXDesc = context_->GetInputDesc(0);
    if (inputXDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "invalid input pointer: x");
        return ge::GRAPH_FAILED;
    }

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
       
        return ge::GRAPH_FAILED;
    }

    if (TilingProcess() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

uint64_t LowerTriangularInverseBaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus LowerTriangularInverseBaseTiling::PostTiling()
{
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(1);
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
        return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

}
}
