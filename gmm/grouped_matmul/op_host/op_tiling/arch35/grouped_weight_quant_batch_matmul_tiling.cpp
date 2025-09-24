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
 * \file grouped_weight_quant_batch_matmul_tiling.cpp
 * \brief
 */
#include "grouped_weight_quant_batch_matmul_tiling.h"

namespace optiling {

bool GroupedWeightQuantBatchMatmulTiling::SetTiling(gert::TilingContext *context)
{
    OP_CHECK_IF(!AnalyzeAttr(context), OP_LOGE(context->GetNodeName(), "Invalid attr param"),
               return false);
    OP_CHECK_IF(!CalcResplitTiling(context),
               OP_LOGE(context->GetNodeName(), "Unable to calculate resplit-tiling"), return false);
    SetBaseTiling();
    SetMatMulTiling();
    SetTilingKey(context);
    OP_CHECK_IF(!SetCustomParam(context),
               OP_LOGE(context->GetNodeName(), "Unable to set custom param"), return false);
    PrintTilingResult(context);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::AnalyzeAttr(const gert::TilingContext *context)
{
    auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is nullptr."), return false);
    coreNum_ = compileInfoPtr->aicNum;

    auto xDesc = context->GetDynamicInputDesc(X_IDX, 0);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context->GetNodeName(), "xDesc is nullptr."), return false);
    xDType_ = xDesc->GetDataType();

    auto wDesc = context->GetDynamicInputDesc(WEIGHT_IDX, 0);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context->GetNodeName(), "wDesc is nullptr."), return false);
    weightDtype_ = wDesc->GetDataType();
    auto wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    if(wFormat == ge::FORMAT_FRACTAL_NZ_C0_16 || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32){
        wFormat = ge::FORMAT_FRACTAL_NZ;
    }
    weightNzFlag_ = wFormat == ge::FORMAT_FRACTAL_NZ;

    auto biasPtr = context->GetDynamicInputTensor(BIAS_IDX, 0);
    hasBias_ = !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);

    auto antiquantScaleDesc = context->GetDynamicInputDesc(ANTIQUANT_SCALE_IDX, 0);
    OP_CHECK_IF(antiquantScaleDesc == nullptr, OP_LOGE(context->GetNodeName(), "antiquantScaleDesc is nullptr."), return false);

    auto antiquantOffsetTensor = context->GetDynamicInputTensor(ANTIQUANT_OFFSET_IDX, 0);
    hasAntiquantOffset_ =
        !(antiquantOffsetTensor == nullptr || antiquantOffsetTensor->GetStorageShape().GetShapeSize() == 0);

    auto attr = context->GetAttrs();
    OP_CHECK_IF(attr == nullptr, OP_LOGE(context->GetNodeName(), "attr is nullptr."), return false);
    const bool *transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_TRANS_W_IDX);
    const bool *transposeXPtr = attr->GetAttrPointer<bool>(ATTR_TRANS_X_IDX);
    const int32_t *groupTypePtr = attr->GetAttrPointer<int32_t>(ATTR_GROUPTYPE_IDX);
    const int64_t *splitItemPtr = attr->GetAttrPointer<int64_t>(ATTR_SPLIT_ITEM_IDX);
    const uint32_t *groupListTypePtr = attr->GetAttrPointer<uint32_t>(ATTR_GROUP_LIST_TYPE_IDX);
    transA_ = transposeXPtr != nullptr ? *transposeXPtr : false;
    transB_ = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    groupType_ = groupTypePtr != nullptr ? static_cast<GroupType>(*groupTypePtr) : GroupType::NO_SPLIT;
    splitItem_ = splitItemPtr != nullptr ? *splitItemPtr : 0;  // 0: 默认split_item
    groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;
    isSingleX_ = (context->GetDynamicInputTensor(X_IDX, 1) == nullptr);
    isSingleWeight_ = (context->GetDynamicInputTensor(WEIGHT_IDX, 1) == nullptr);
    // 2: when x is multi-tensor, y is single-tensor; 3: when x is single-tensor, y is single-tensor
    isSingleY_ = (splitItem_ == 2 || splitItem_ == 3);

    OP_CHECK_IF(!CheckAttr(context), OP_LOGE(context->GetNodeName(), "Invalid attr param"),
               return false);
    OP_CHECK_IF(!SetShapeListSplitMSingleXSingleWeightSingleY(context),
               OP_LOGE(context->GetNodeName(), "Unable to get shape list"), return false);
    OP_CHECK_IF(!SetAntiquantGroupSize(context),
               OP_LOGE(context->GetNodeName(), "Unable to get antiquant groupSize"), return false);
    if(weightDtype_ == ge::DT_FLOAT){
        weightDtype_ = ge::DT_FLOAT4_E2M1;
        nSize_ = static_cast<uint64_t>(8) * nSize_; //一个float32表示8个fp4,设置为正确shape
    }
    PrintInputParam(context);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CalcResplitTiling(const gert::TilingContext *context)
{
    uint64_t c0Size;
    OP_CHECK_IF(
        !GetC0Size(context, xDType_, c0Size) || (weightNzFlag_ && !transB_ && nSize_ % c0Size > 0),
        OP_LOGE(
            context->GetNodeName(),
            "Invalid C0 size[%lu], expect greater than 0 and divisible by N[%lu] when weight format is FRACTAL_NZ",
            c0Size, nSize_),
        return false);
    OP_CHECK_IF(
        coreNum_ <= 0,
        OP_LOGE(context->GetNodeName(), "Invalid core num[%u], expect greater than 0", coreNum_),
        return false);

    cubeBlockDimN_ = static_cast<uint8_t>(coreNum_);
    if (nSize_ % (coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N)) == 0UL) {
        resplitParam_.mainBlockSize = BASIC_BLOCK_BASE_N;
        resplitParam_.mainBlockCount = nSize_ / (coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N));
    } else if (nSize_ >= coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N_MIN)) {
        // 该场景下可以保证分满核且尾块在128~256之间
        CalcFullBlockDimResplitTiling(c0Size);
    } else {
        // N <= 4096场景，优先保证单核尾块大于128，可能无法分满核
        CalcNoFullBlockDimResplitTiling(c0Size);
    }
    OP_CHECK_IF(!CheckResplitTilingResult(context),
               OP_LOGE(context->GetNodeName(), "Invalid resplit tiling result"), return false);
    return true;
}

void GroupedWeightQuantBatchMatmulTiling::SetBaseTiling()
{
    tilingData_.gmmWeightQuantParam.set_groupNum(groupNum_);
    tilingData_.gmmWeightQuantParam.set_coreNum(coreNum_);
    tilingData_.gmmWeightQuantParam.set_kSize(kSize_);
    tilingData_.gmmWeightQuantParam.set_nSize(nSizeOri_);
    tilingData_.gmmWeightQuantParam.set_singleX(static_cast<uint8_t>(isSingleX_));
    tilingData_.gmmWeightQuantParam.set_singleWeight(static_cast<uint8_t>(isSingleWeight_));
    tilingData_.gmmWeightQuantParam.set_singleY(static_cast<uint8_t>(isSingleY_));
    tilingData_.gmmWeightQuantParam.set_groupType(static_cast<int8_t>(groupType_));
    tilingData_.gmmWeightQuantParam.set_groupListType(static_cast<uint8_t>(groupListType_));
    tilingData_.gmmWeightQuantParam.set_hasBias(static_cast<uint8_t>(hasBias_));
    tilingData_.gmmWeightQuantParam.set_cubeBlockDimN(cubeBlockDimN_);
    tilingData_.gmmWeightQuantParam.set_groupSize(groupSize_);
    tilingData_.gmmWeightQuantParam.set_mainBlockSize(resplitParam_.mainBlockSize);
    tilingData_.gmmWeightQuantParam.set_mainBlockCount(resplitParam_.mainBlockCount * coreNum_);
    tilingData_.gmmWeightQuantParam.set_firstTailBlockSize(resplitParam_.firstTailBlockSize);
    tilingData_.gmmWeightQuantParam.set_secondTailBlockSize(resplitParam_.secondTailBlockSize);
    tilingData_.gmmWeightQuantParam.set_firstTailBlockCount(resplitParam_.firstTailBlockCount);
    tilingData_.gmmWeightQuantParam.set_secondTailBlockCount(resplitParam_.secondTailBlockCount);
    tilingData_.gmmArray.set_mList(mList_);
    tilingData_.gmmArray.set_kList(kList_);
    tilingData_.gmmArray.set_nList(nList_);
}

void GroupedWeightQuantBatchMatmulTiling::SetMatMulTiling()
{
    tilingData_.mmTilingData.set_baseM(BASIC_BLOCK_BASE_M);
    tilingData_.mmTilingData.set_singleCoreM(BASIC_BLOCK_BASE_M);
    tilingData_.mmTilingData.set_isBias(static_cast<int32_t>(hasBias_));
    tilingData_.mmTilingData.set_M(mSize_);
    tilingData_.mmTilingData.set_N(nSize_);
    tilingData_.mmTilingData.set_Ka(kSize_);
    tilingData_.mmTilingData.set_Kb(kSize_);
    tilingData_.mmTilingData.set_singleCoreN(BASIC_BLOCK_BASE_N);
    tilingData_.mmTilingData.set_singleCoreK(kSize_);
    tilingData_.mmTilingData.set_dbL0A(BUFFER_NUM_2);
    tilingData_.mmTilingData.set_dbL0B(BUFFER_NUM_2);
    tilingData_.mmTilingData.set_dbL0C(1);
    tilingData_.mmTilingData.set_shareL0CSize(BASIC_BLOCK_BASE_M * BASIC_BLOCK_BASE_N * sizeof(float));

    tilingData_.mmTilingData.set_baseN(BASIC_BLOCK_BASE_N);
    tilingData_.mmTilingData.set_baseK(BASIC_BLOCK_BASE_K);
    tilingData_.mmTilingData.set_stepKa(STEP_K_4);
    tilingData_.mmTilingData.set_stepKb(STEP_K_4);
    tilingData_.mmTilingData.set_depthA1(DEPTH_8);
    tilingData_.mmTilingData.set_depthB1(DEPTH_8);
    tilingData_.mmTilingData.set_stepM(1);
    tilingData_.mmTilingData.set_stepN(1);
    tilingData_.mmTilingData.set_usedCoreNum(coreNum_);

    if (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4) {
        // 2含义：S8S4场景，MAD采用S8类型，baseK需要放大2倍
        tilingData_.mmTilingData.set_baseK(BASIC_BLOCK_BASE_K * 2);
        // A8W4场景在UB中处理bias，mm api默认无bias
        tilingData_.mmTilingData.set_isBias(0);
    } else if (hasBias_) {
        tilingData_.mmTilingData.set_baseM(BASIC_BLOCK_BASE_M_WITH_BIAS);
        tilingData_.mmTilingData.set_singleCoreM(BASIC_BLOCK_BASE_M_WITH_BIAS);
    }
}

void GroupedWeightQuantBatchMatmulTiling::SetTilingKey(gert::TilingContext *context)
{
    constexpr uint8_t DECIMAL = 10U;
    // 平台类型占2位(平台大类， 平台小类)，平台大类在高位，需要乘10
    tilingKeyConfig_.socVersionType = static_cast<uint8_t>(SocVersionType::SUPPORT_L1_TO_BT_BF16) * DECIMAL;
    tilingKeyConfig_.quantizationScenario = static_cast<uint8_t>(QuantizationScenario::DEFAULT);
    // 算法类型占2位(算法大类，算法小类)，算法大类在高位，需要乘10
    tilingKeyConfig_.algorithm = static_cast<uint8_t>(OptimizationAlgorithmCategory::VECTOR_ANTIQUANT) * DECIMAL +
                                 static_cast<uint8_t>(OptimizationAlgorithmSubCategory::N_FIRST_TAIL_RESPLIT);

    tilingKeyConfig_.transposeSituation = (static_cast<uint8_t>(transA_) << 1) | static_cast<uint8_t>(transB_);

    auto antiquantScaleDesc = context->GetDynamicInputDesc(ANTIQUANT_SCALE_IDX, 0);
    if (antiquantScaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0) {
        tilingKeyConfig_.antiquantType = static_cast<uint8_t>(QuantType::MX);
    } else {
        tilingKeyConfig_.antiquantType = static_cast<uint8_t>(QuantType::PER_CHANNEL);
    }

    tilingKeyConfig_.quantType = static_cast<uint8_t>(QuantType::NONE);
    tilingKeyConfig_.optionInputSituation = static_cast<uint8_t>(hasAntiquantOffset_) << 1;
    tilingKeyConfig_.weightFormat =
        weightNzFlag_ ? static_cast<uint8_t>(WeightFormat::FRACTAL_NZ) : static_cast<uint8_t>(WeightFormat::ND);
    tilingKeyConfig_.templateCustom = static_cast<uint8_t>(Mte2Configuration::MTE2_INNER_SIZE_256_BUF_NUM_4);
    if (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4) {
        tilingKeyConfig_.templateCustom = static_cast<uint8_t>(Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT);
    }
    tilingKeyConfig_.apiConstexpr = 0U;
    context->SetTilingKey(tilingKeyConfig_.GenTilingKey());
}

bool GroupedWeightQuantBatchMatmulTiling::SetCustomParam(gert::TilingContext *context)
{
    size_t *workspaces = context->GetWorkspaceSizes(1);    // get second variable
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context->GetNodeName(), "workspaces is nullptr."), return false);  // check workspaces is not null
    workspaces[0] = 16777216U;                             // 16 * 1024 * 1024: default workspace size

    context->SetBlockDim(coreNum_);
    OP_CHECK_IF(context->GetRawTilingData() == nullptr, OP_LOGE(context->GetNodeName(), "RawTilingData is nullptr."), return false);
    tilingData_.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckAttr(const gert::TilingContext *context) const
{
    OP_CHECK_IF(
        coreNum_ <= 0,
        OP_LOGE(context->GetNodeName(), "Invalid coreNum[%u], expect greater than 0", coreNum_),
        return false);
    OP_CHECK_IF(groupType_ != GroupType::SPLIT_M || !(isSingleX_ && isSingleWeight_ && isSingleY_),
               OP_LOGE(context->GetNodeName(),
                                           "Only support groupType 0 (split-m mode), single-single-single mode, actual "
                                           "groupType: %d, singleX: %s, singleW: %s, singleY: %s",
                                           static_cast<int8_t>(groupType_), isSingleX_ ? "true" : "false",
                                           isSingleWeight_ ? "true" : "false", isSingleY_ ? "true" : "false"),
               return false);
    OP_CHECK_IF(weightNzFlag_ && transB_,
               OP_LOGE(context->GetNodeName(),
                                           "transposed weight is unsupported when weight format is FRACTAL_NZ"),
               return false);
    OP_CHECK_IF(!weightNzFlag_ && !transB_,
               OP_LOGE(context->GetNodeName(),
                                           "untransposed weight is unsupported when weight format is ND"),
               return false);
    OP_CHECK_IF(xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4 && hasAntiquantOffset_,
               OP_LOGE(context->GetNodeName(), "antiquantOffset is unsupported for A8W4"),
               return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::SetShapeListSplitMSingleXSingleWeightSingleY(
    const gert::TilingContext *context)
{
    auto xTensor = context->GetDynamicInputTensor(X_IDX, 0);  // 0: get first tensor
    OP_CHECK_IF(xTensor == nullptr, OP_LOGE(context->GetNodeName(), "xTensor is nullptr."), return false);
    gert::Shape xShape = xTensor->GetStorageShape();

    auto wTensor = context->GetDynamicInputTensor(WEIGHT_IDX, 0);  // 0: get first tensor
    OP_CHECK_IF(wTensor == nullptr, OP_LOGE(context->GetNodeName(), "wTensor is nullptr."), return false);
    gert::Shape wShape = wTensor->GetStorageShape();

    groupNum_ = static_cast<int32_t>(wShape.GetDim(0));
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    OP_CHECK_IF(weightNzFlag_ && wDimNum < GroupedMatmul::MIN_NZ_DIM,
               OP_LOGE(
                   context->GetNodeName(),
                   "Invalid weight dimension for format FRACTAL_NZ, expect at least 4, actual %u", wDimNum),
               return false);
    OP_CHECK_IF(
        !weightNzFlag_ && wDimNum < GroupedMatmul::MIN_ND_DIM,
        OP_LOGE(context->GetNodeName(),
                                    "Invalid weight dimension for format ND, expect at least 2, actual %u", wDimNum),
        return false);
    uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
    OP_CHECK_IF(xDimNum < GroupedMatmul::MIN_ND_DIM,
               OP_LOGE(context->GetNodeName(),
                                           "Invalid x dimension for format ND, expect at least 2, actual %u", xDimNum),
               return false);
    mSize_ = transA_ ? xShape.GetDim(1) : xShape.GetDim(0);
    kSize_ = transA_ ? xShape.GetDim(0) : xShape.GetDim(xDimNum - 1);
    // -1含义为(K, N)场景N索引，-2含义为(N, K)场景N索引
    nSize_ = transB_ ? wShape.GetDim(wDimNum - 2) : wShape.GetDim(wDimNum - 1);
    nSizeOri_ = nSize_;
    if (weightNzFlag_) {
        // 非转置NZ排布(N1, K1, K0, N0), 转置NZ排布(K1, N1, N0, K0)
        // -3含义：转置N1索引；-2含义：转置N0索引
        nSize_ = transB_ ? wShape.GetDim(wDimNum - 3) * wShape.GetDim(wDimNum - 2)
                         // -4含义：非转置N1索引；-1含义：非转置N0索引
                         : wShape.GetDim(wDimNum - 4) * wShape.GetDim(wDimNum - 1);
        const gert::StorageShape *yShapePtr = context->GetOutputShape(0);
        OP_CHECK_IF(yShapePtr == nullptr, OP_LOGE(context->GetNodeName(), "yShapePtr is nullptr."), return false);
        const gert::Shape &yShape = yShapePtr->GetOriginShape();
        nSizeOri_ = yShape.GetDim(static_cast<uint32_t>(yShape.GetDimNum()) - 1);
    }
    OP_CHECK_IF(mSize_ <= 0 || kSize_ <= 0 || nSize_ <= 0,
               OP_LOGE(context->GetNodeName(),
                                           "Invalid mSize[%lu] kSize[%lu] or nSize[%lu], expect all greater than 0",
                                           mSize_, kSize_, nSize_),
               return false);

    kList_[0] = static_cast<int32_t>(kSize_);
    nList_[0] = static_cast<int32_t>(nSizeOri_);
    mList_[0] = -1;
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::SetAntiquantGroupSize(const gert::TilingContext *context)
{
    auto antiquantScale = context->GetDynamicInputTensor(ANTIQUANT_SCALE_IDX, 0);
    OP_CHECK_IF(antiquantScale == nullptr, OP_LOGE(context->GetNodeName(), "antiquantScale is nullptr."), return false);
    auto antiquantScaleShape = antiquantScale->GetStorageShape();
    int64_t antiquantScaleDimNum = antiquantScaleShape.GetDimNum();
    // 3含义：单场景antiquantScale维度在切M时是(g, N, K)或(g, K, N)
    if (antiquantScaleDimNum == 3) {
        // 2含义：(g, K, N)格式K轴索引
        int64_t groupNum = transB_ ? antiquantScaleShape.GetDim(antiquantScaleDimNum - 1)
                                   : antiquantScaleShape.GetDim(antiquantScaleDimNum - 2);
        OP_CHECK_IF(groupNum <= 0 || kSize_ % groupNum > 0,
                   OP_LOGE(
                       context->GetNodeName(),
                       "Invalid groupNum[%ld], expect greater than 0 and divisible by kSize[%lu]", groupNum, kSize_),
                   return false);
        // GMM伪量化场景支持K=groupSize
        groupSize_ = groupNum > 0 ? kSize_ / static_cast<uint64_t>(groupNum) : 0;
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::GetC0Size(const gert::TilingContext *context, ge::DataType dtype,
                                                    uint64_t &c0Size) const
{
    if (dtype == ge::DT_INT4) {
        c0Size = AscendC::ONE_BLK_SIZE + AscendC::ONE_BLK_SIZE;
    } else {
        int64_t dtypeSize = ge::GetSizeByDataType(dtype);
        OP_CHECK_IF(dtypeSize <= 0,
                   OP_LOGE(context->GetNodeName(), "Invalid dtypeSize[%ld], expect greater than 0",
                                               dtypeSize),
                   return false);
        if (dtypeSize > 0) {
            c0Size = AscendC::ONE_BLK_SIZE / dtypeSize;
        }
    }
    return true;
}

void GroupedWeightQuantBatchMatmulTiling::CalcFullBlockDimResplitTiling(uint64_t c0Size)
{
    resplitParam_.mainBlockSize = BASIC_BLOCK_BASE_N;
    resplitParam_.mainBlockCount = 0UL;
    if (nSize_ / (coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N)) > 1UL) {
        resplitParam_.mainBlockCount = nSize_ / (coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N)) - 1UL;
    }
    uint64_t tailSizeOri = nSize_ - resplitParam_.mainBlockCount * resplitParam_.mainBlockSize * coreNum_;
    uint64_t tailSize = tailSizeOri;
    if (weightNzFlag_ && c0Size != 0UL) {
        tailSize = tailSizeOri / c0Size;
    }
    if (tailSizeOri > coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N)) {
        // 2含义：当尾块大于核数*256，除以2倍核数以保证单核尾块大小小于256
        constexpr uint64_t resplitFactor = 2;
        resplitParam_.firstTailBlockSize = static_cast<uint16_t>(tailSize / (coreNum_ * resplitFactor));
        resplitParam_.secondTailBlockSize = static_cast<uint16_t>(resplitParam_.firstTailBlockSize + 1U);
        resplitParam_.secondTailBlockCount = static_cast<uint16_t>(tailSize % (coreNum_ * resplitFactor));
        resplitParam_.firstTailBlockCount =
            static_cast<uint16_t>(coreNum_ * resplitFactor - resplitParam_.secondTailBlockCount);
    } else {
        resplitParam_.firstTailBlockSize = static_cast<uint16_t>(tailSize / coreNum_);
        resplitParam_.secondTailBlockSize = static_cast<uint16_t>(resplitParam_.firstTailBlockSize + 1U);
        resplitParam_.secondTailBlockCount = static_cast<uint16_t>(tailSize % coreNum_);
        resplitParam_.firstTailBlockCount = static_cast<uint16_t>(coreNum_ - resplitParam_.secondTailBlockCount);
    }
    if (weightNzFlag_) {
        resplitParam_.firstTailBlockSize *= static_cast<uint16_t>(c0Size);
        resplitParam_.secondTailBlockSize *= static_cast<uint16_t>(c0Size);
    }
}

void GroupedWeightQuantBatchMatmulTiling::CalcNoFullBlockDimResplitTiling(uint64_t c0Size)
{
    resplitParam_.mainBlockSize = BASIC_BLOCK_BASE_N;
    resplitParam_.mainBlockCount = 0UL;
    uint64_t taskNum = std::max(1UL, nSize_ / BASIC_BLOCK_BASE_N_MIN);  // 实际任务数，必然小于核数
    cubeBlockDimN_ = static_cast<uint8_t>(taskNum);
    if (weightNzFlag_ && c0Size != 0UL && taskNum != 0UL) {
        resplitParam_.firstTailBlockSize = static_cast<uint16_t>(nSize_ / c0Size / taskNum);
        resplitParam_.secondTailBlockSize = static_cast<uint16_t>(resplitParam_.firstTailBlockSize + 1U);
        resplitParam_.secondTailBlockCount = static_cast<uint16_t>(nSize_ / c0Size % taskNum);
        resplitParam_.firstTailBlockCount = static_cast<uint16_t>(taskNum - resplitParam_.secondTailBlockCount);
        resplitParam_.firstTailBlockSize *= static_cast<uint16_t>(c0Size);
        resplitParam_.secondTailBlockSize *= static_cast<uint16_t>(c0Size);
    } else if (taskNum != 0UL) {
        resplitParam_.firstTailBlockSize = static_cast<uint16_t>(nSize_ / taskNum);
        resplitParam_.secondTailBlockSize = static_cast<uint16_t>(resplitParam_.firstTailBlockSize + 1U);
        resplitParam_.secondTailBlockCount = static_cast<uint16_t>(nSize_ % taskNum);
        resplitParam_.firstTailBlockCount = static_cast<uint16_t>(taskNum - resplitParam_.secondTailBlockCount);
    }
}

bool GroupedWeightQuantBatchMatmulTiling::CheckResplitTilingResult(const gert::TilingContext *context) const
{
    OP_CHECK_IF(
        nSize_ != static_cast<uint64_t>(resplitParam_.mainBlockCount) * coreNum_ * resplitParam_.mainBlockSize +
                      resplitParam_.firstTailBlockCount * resplitParam_.firstTailBlockSize +
                      resplitParam_.secondTailBlockCount * resplitParam_.secondTailBlockSize,
        OP_LOGE(
            context->GetNodeName(),
            "Invalid resplit tiling result, expect nSize[%lu] == mainBlockCount[%lu] x coreNum[%u] x mainBlockSize[%u] "
            "+ firstTailBlockCount[%u] x firstTailBlockSize[%u] + secondTailBlockCount[%u] x secondTailBlockSize[%u]",
            nSize_, resplitParam_.mainBlockCount, coreNum_, resplitParam_.mainBlockSize,
            resplitParam_.firstTailBlockCount, resplitParam_.firstTailBlockSize, resplitParam_.secondTailBlockCount,
            resplitParam_.secondTailBlockSize),
        return false);
    OP_CHECK_IF(
        nSize_ >= static_cast<uint64_t>(coreNum_) * BASIC_BLOCK_BASE_N_MIN &&
            (resplitParam_.firstTailBlockCount + resplitParam_.secondTailBlockCount) % coreNum_ > 0,
        OP_LOGE(context->GetNodeName(),
                                    "Invalid resplit tiling result, expect core num [%u] is divisible by "
                                    "(firstTailBlockCount[%u] + secondTailBlockCount[%u])",
                                    coreNum_, resplitParam_.firstTailBlockCount, resplitParam_.secondTailBlockCount),
        return false);
    OP_CHECK_IF(
        nSize_ >= static_cast<int64_t>(BASIC_BLOCK_BASE_N_MIN) && resplitParam_.firstTailBlockCount > 0 &&
            (resplitParam_.firstTailBlockSize < BASIC_BLOCK_BASE_N_MIN ||
             resplitParam_.firstTailBlockSize > BASIC_BLOCK_BASE_N),
        OP_LOGE(context->GetNodeName(),
                                    "Invalid resplit tiling result, expect [%u] <= firstTailBlockSize [%u] <= [%u] ",
                                    BASIC_BLOCK_BASE_N_MIN, resplitParam_.firstTailBlockSize, BASIC_BLOCK_BASE_N),
        return false);
    OP_CHECK_IF(
        nSize_ >= static_cast<uint64_t>(BASIC_BLOCK_BASE_N_MIN) && resplitParam_.secondTailBlockCount > 0 &&
            (resplitParam_.secondTailBlockSize < BASIC_BLOCK_BASE_N_MIN ||
             resplitParam_.secondTailBlockSize > BASIC_BLOCK_BASE_N),
        OP_LOGE(context->GetNodeName(),
                                    "Invalid resplit tiling result, expect [%u] <= secondTailBlockSize [%u] <= [%u] ",
                                    BASIC_BLOCK_BASE_N_MIN, resplitParam_.secondTailBlockSize, BASIC_BLOCK_BASE_N),
        return false);
    return true;
}

void GroupedWeightQuantBatchMatmulTiling::PrintInputParam(const gert::TilingContext *context) const
{
    OP_LOGI(context->GetNodeName(),
              "Input params: coreNum: %u, gmm groupNum: %u, groupType: %d, groupListType: %u, splitItem: %ld, "
              "antiquant-groupSize: %u, mSize: %lu, kSize: %lu, nSize: %lu, nSizeOri: %lu, transA: %s, transB: %s, "
              "isSingleX: %s, isSingleWeight: %s, isSingleY: %s, hasBias: %s, weightNzFlag: %s, hasAntiquantOffset: "
              "%s, xDtype: %s, weightDtype: %s",
              coreNum_, groupNum_, static_cast<int8_t>(groupType_), groupListType_, splitItem_, groupSize_, mSize_,
              kSize_, nSize_, nSizeOri_, transA_ ? "true" : "false", transB_ ? "true" : "false",
              isSingleX_ ? "true" : "false", isSingleWeight_ ? "true" : "false", isSingleY_ ? "true" : "false",
              hasBias_ ? "true" : "false", weightNzFlag_ ? "true" : "false", hasAntiquantOffset_ ? "true" : "false",
              ge::TypeUtils::DataTypeToSerialString(xDType_).c_str(),
              ge::TypeUtils::DataTypeToSerialString(weightDtype_).c_str());
}

void GroupedWeightQuantBatchMatmulTiling::PrintTilingResult(const gert::TilingContext *context)
{
    OP_LOGI(
        context->GetNodeName(),
        "Tiling result: groupNum: %u, coreNum: %u, kSize: %lu, nSize: %lu, singleX: %u, singleWeight: %u, singleY: %u, "
        "groupType: %d, groupListType: %u, hasBias: %u, groupSize: %u, mainBlockSize: %u, mainBlockCount: %lu, "
        "firstTailBlockSize: %u, secondTailBlockSize: %u, firstTailBlockCount: %u, secondTailBlockCount: %u",
        tilingData_.gmmWeightQuantParam.get_groupNum(), tilingData_.gmmWeightQuantParam.get_coreNum(),
        tilingData_.gmmWeightQuantParam.get_kSize(), tilingData_.gmmWeightQuantParam.get_nSize(),
        tilingData_.gmmWeightQuantParam.get_singleX(), tilingData_.gmmWeightQuantParam.get_singleWeight(),
        tilingData_.gmmWeightQuantParam.get_singleY(), tilingData_.gmmWeightQuantParam.get_groupType(),
        tilingData_.gmmWeightQuantParam.get_groupListType(), tilingData_.gmmWeightQuantParam.get_hasBias(),
        tilingData_.gmmWeightQuantParam.get_groupSize(), tilingData_.gmmWeightQuantParam.get_mainBlockSize(),
        tilingData_.gmmWeightQuantParam.get_mainBlockCount(), tilingData_.gmmWeightQuantParam.get_firstTailBlockSize(),
        tilingData_.gmmWeightQuantParam.get_secondTailBlockSize(),
        tilingData_.gmmWeightQuantParam.get_firstTailBlockCount(),
        tilingData_.gmmWeightQuantParam.get_secondTailBlockCount());
}
}  // namespace optiling