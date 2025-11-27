/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_infershape_weight_quant_checker.cpp
 * \brief
 */
#include "grouped_matmul_infershape_weight_quant_checker.h"
#include "grouped_matmul_infershape_common_util.h"

namespace ops {
static const std::unordered_set<ge::DataType> X_TYPE_SUPPORT_SET = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN,
                                                                    ge::DT_INT8};
static const std::unordered_set<ge::DataType> WEIGHT_TYPE_SUPPORT_SET = {
    ge::DT_INT8,        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8,
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT,         ge::DT_INT4,        ge::DT_INT32};
static const std::map<ge::DataType, std::unordered_set<ge::DataType>> BIAS_TYPE_SUPPORT_MAP = {
    {ge::DT_FLOAT16, {ge::DT_FLOAT16}},
    {ge::DT_BF16, {ge::DT_BF16, ge::DT_FLOAT}},
    {ge::DT_INT8, {ge::DT_FLOAT}},
    {ge::DT_FLOAT8_E4M3FN, {ge::DT_BF16, ge::DT_FLOAT16}}};
static const std::unordered_set<ge::DataType> FP8_SUPPORT_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8};
const int64_t UNKNOWN_SHAPE_VALUE = -1;
const int64_t SHAPE_UNKNOWN_DIM_NUM = -2;
const int64_t B4_NUMS_IN_B32 = 8;

static bool inline IsNonEmpty(const gert::Shape *shape)
{
    return (shape != nullptr && !(shape->GetDimNum() == 1 && shape->GetDim(0) == 0));
}

bool GroupedMatmulWeightQuantChecker::IsA16MxFp4NZ(const ge::DataType &xDtype, const ge::DataType &weightDtype) const
{
    return (xDtype == ge::DT_FLOAT16 || xDtype == ge::DT_BF16) &&
           (weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT);
}

bool GroupedMatmulWeightQuantChecker::IsMxA8W4NZ(const ge::DataType &xDtype, const ge::DataType &weightDtype) const
{
    return xDtype == ge::DT_FLOAT8_E4M3FN && (weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT);
}

bool GroupedMatmulWeightQuantChecker::IsS8S4NZ(const ge::DataType &xDtype, const ge::DataType &weightDtype) const
{
    return xDtype == ge::DT_INT8 && (weightDtype == ge::DT_INT4 || weightDtype == ge::DT_INT32);
}

bool GroupedMatmulWeightQuantChecker::IsA16W8(const ge::DataType &xDtype, const ge::DataType &weightDtype) const
{
    return (xDtype == ge::DT_FLOAT16 || xDtype == ge::DT_BF16) && weightDtype == ge::DT_INT8;
}

bool GroupedMatmulWeightQuantChecker::IsA8W4(const ge::DataType &xDtype, const ge::DataType &weightDtype) const
{
    return IsMxA8W4NZ(xDtype, weightDtype) || IsS8S4NZ(xDtype, weightDtype);
}

ge::graphStatus GroupedMatmulWeightQuantChecker::GetXAndWeightDimValue(const gert::InferShapeContext *context,
                                                                       const GMMAttrs &gmmAttrs)
{
    auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_IF(!(IsNonEmpty(xShape) && IsNonEmpty(weightShape)),
                OP_LOGE(context->GetNodeName(), "The 1st tensor of tensor list x and weight cannot be empty."),
                return ge::GRAPH_FAILED);
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    auto weightDtype = weightDesc->GetDataType();
    xdimNum_ = xShape->GetDimNum();
    weightdimNum_ = weightShape->GetDimNum();
    OP_CHECK_IF(xdimNum_ < GMM_MIN_FM_DIM,
                OP_LOGE(context->GetNodeName(),
                        "The dim num of x's 1st tensor should be greater than 1, but it is [%zu].", xdimNum_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightdimNum_ < GMM_MIN_WEIGHT_DIM,
                OP_LOGE(context->GetNodeName(),
                        "The dim num of weight's 1st tensor should be greater than 1, but it is [%zu].", weightdimNum_),
                return ge::GRAPH_FAILED);

    xMDim_ = gmmAttrs.transposeX ? xShape->GetDim(xdimNum_ - 1) : xShape->GetDim(xdimNum_ - PENULTIMATE_DIM);
    xKDim_ = gmmAttrs.transposeX ? xShape->GetDim(xdimNum_ - PENULTIMATE_DIM) : xShape->GetDim(xdimNum_ - 1);
    weightKDim_ = gmmAttrs.transposeWeight ? weightShape->GetDim(weightdimNum_ - 1) :
                                             weightShape->GetDim(weightdimNum_ - PENULTIMATE_DIM);
    weightNDim_ = gmmAttrs.transposeWeight ? weightShape->GetDim(weightdimNum_ - PENULTIMATE_DIM) :
                                             weightShape->GetDim(weightdimNum_ - 1);
    // float表示8个float4_e2m1，推导shape时，尾轴扩大8倍
    if (weightDtype == ge::DT_FLOAT && weightKDim_ > 0 && xKDim_ > 0) {
        bool transWeightFp32 = false;
        if (!gmmAttrs.transposeWeight) {
            transWeightFp32 = weightKDim_ * B4_NUMS_IN_B32 == xKDim_;
        }
        if (gmmAttrs.transposeWeight || transWeightFp32) {
            weightKDim_ = weightKDim_ * B4_NUMS_IN_B32;
        } else {
            weightNDim_ = weightNDim_ * B4_NUMS_IN_B32;
        }
    } else if (weightDtype == ge::DT_INT32) {
        // 一个int32表示8个int4
        weightNDim_ = weightNDim_ * B4_NUMS_IN_B32;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeForXAndWeight(const gert::InferShapeContext *context,
                                                                         const GMMAttrs &gmmAttrs) const
{
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    ge::DataType weightDtype = weightDesc->GetDataType();
    OP_CHECK_IF(xKDim_ != weightKDim_,
                OP_LOGE(context->GetNodeName(),
                        "The k dim of x should be equal to the k dim of weight, "
                        "but x's k is [%ld] and weight's k is [%ld].",
                        xKDim_, weightKDim_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightNDim_ <= 0,
                OP_LOGE(context->GetNodeName(), "The n dim value should be positive, but the actual value is [%ld].",
                        weightNDim_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightKDim_ <= 0,
                OP_LOGE(context->GetNodeName(), "The k dim value should be positive, but the actual value is [%ld].",
                        weightKDim_),
                return ge::GRAPH_FAILED);
    if (weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT || weightDtype == ge::DT_INT4 ||
        weightDtype == ge::DT_INT32) {
        OP_CHECK_IF(
            !((weightNDim_ % GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0) &&
              (weightKDim_ % GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0)),
            OP_LOGE(context->GetNodeName(),
                    "The value of dim n, k should be an integer multiple of [%ld], but actual n is [%ld], k is [%ld].",
                    GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT, weightNDim_, weightKDim_),
            return ge::GRAPH_FAILED);
    }
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        OP_CHECK_IF(xdimNum_ != GMM_MIN_FM_DIM || weightdimNum_ != GMM_SPLIT_M_SINGLE_WEIGHT_DIM,
                    OP_LOGE(context->GetNodeName(),
                            "When split m, x dim num should be 2, weight dim num should be 3, "
                            "but the actual x dim num is [%zu], actual weight dim num is [%zu].",
                            xdimNum_, weightdimNum_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(weightShape->GetDim(0) != groupNum_,
                    OP_LOGE(context->GetNodeName(),
                            "When split m, 1st dim value of weight should be g, "
                            "which is [%ld], but the actual value is [%ld].",
                            groupNum_, weightShape->GetDim(0)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckTensorDimEqualOne(const gert::InferShapeContext *context,
                                                                        const gert::Shape *shape,
                                                                        const std::string paramName,
                                                                        const size_t index) const
{
    OP_CHECK_IF(
        shape == nullptr,
        OP_LOGE(context->GetNodeName(), "%s Shape[%lu] is null, which is not supported.", paramName.c_str(), index),
        return ge::GRAPH_FAILED);
    size_t dimNum = shape->GetDimNum();
    OP_CHECK_IF(
        dimNum != 1,
        OP_LOGE(context->GetNodeName(), "%s[%lu] dimNum is %lu, but only support 1.", paramName.c_str(), index, dimNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckDimNumNoSplit(const gert::InferShapeContext *context,
                                                                    const GMMInputParamsInfo &paramsInputInfo) const
{
    const size_t &tensorListLength = paramsInputInfo.numX;
    auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    // check dimension
    for (size_t i = 0; i < tensorListLength; ++i) {
        auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, i);
        OP_CHECK_IF(xShape == nullptr, OP_LOGE(context->GetNodeName(), "x[%lu] is null, which is not supported.", i),
                    return ge::GRAPH_FAILED);
        wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
        size_t weightDimNum = wShape->GetDimNum();
        OP_CHECK_IF(weightDimNum != GMM_SEPARATED_WEIGHT_DIM,
                    OP_LOGE(context->GetNodeName(),
                            "weight[%lu] dimNum is %lu, but only support 2 when weight separated.", i, weightDimNum),
                    return ge::GRAPH_FAILED);
        // 校验 x 中每个tensor的维度必须在[2,6]之间
        size_t xDimNum = xShape->GetDimNum();
        OP_CHECK_IF(xDimNum > GMM_MAX_FM_DIM || xDimNum < GMM_MIN_FM_DIM,
                    OP_LOGE(context->GetNodeName(), "x[%lu] dimNum is %lu, but only support 2-6.", i, xDimNum),
                    return ge::GRAPH_FAILED);
        // 检测 bias antiquantScale antiquantOffset 的每个tensor的dim都需要为1
        if (paramsInputInfo.numBias != 0) {
            auto baisShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, i);
            OP_CHECK_IF(CheckTensorDimEqualOne(context, baisShape, "bais", i) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "CheckTensorDimEqualOne is failed."), return ge::GRAPH_FAILED);
        }
        if (paramsInputInfo.numAntiquantOffset != 0) {
            auto antiquantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, i);
            OP_CHECK_IF(CheckTensorDimEqualOne(context, antiquantOffsetShape, "antiquantOffset", i) !=
                            ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "CheckTensorDimEqualOne is failed."), return ge::GRAPH_FAILED);
        }
        auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, i);
        OP_CHECK_IF(CheckTensorDimEqualOne(context, antiquantScaleShape, "antiquantScale", i) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckTensorDimEqualOne is failed."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckXWeightYGroupSizeMultiSenario(
                                                                    const gert::InferShapeContext *context,
                                                                    const GMMInputParamsInfo &paramsInputInfo) const
{
    const size_t &xSize = paramsInputInfo.numX;
    const size_t &weightSize = paramsInputInfo.numWeight;
    size_t numY = context->GetComputeNodeOutputNum();
    // check group size
    OP_CHECK_IF(xSize != numY,
                OP_LOGE(context->GetNodeName(), "When y is separated, size of x %lu should equal to size of y %lu.",
                        xSize, numY),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xSize != weightSize,
                OP_LOGE(context->GetNodeName(),
                        "When weight is separated, size of w %lu should equal to size of x %lu.", weightSize, xSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckTensorNDimMultiSenario(const gert::InferShapeContext *context,
                                                                             const GMMInputParamsInfo &paramsInputInfo,
                                                                             const size_t wNDimIdx,
                                                                             const int64_t weightNDimValue,
                                                                             const size_t index) const
{
    // 检验weight的n轴和antiquantScale的n轴一致
    auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, index);
    OP_CHECK_NULL_WITH_CONTEXT(context, antiquantScaleShape);
    int64_t antiquantScaleNDim = antiquantScaleShape->GetDim(0);
    OP_CHECK_IF(antiquantScaleNDim != weightNDimValue,
                OP_LOGE(context->GetNodeName(),
                        "weight[%lu] dim %lu value %ld should equal to antiquantScale[%lu] dim 0 value %ld.", index,
                        wNDimIdx, weightNDimValue, index, antiquantScaleNDim),
                return ge::GRAPH_FAILED);

    if (paramsInputInfo.numBias != 0) {
        // 检验weigh的n轴和bias的n轴一致
        auto baisShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, index);
        OP_CHECK_NULL_WITH_CONTEXT(context, baisShape);
        int64_t baisNDim = baisShape->GetDim(0);
        OP_CHECK_IF(baisNDim != weightNDimValue,
                    OP_LOGE(context->GetNodeName(),
                            "weight[%lu] dim %lu value %ld should equal to bais[%lu] dim 0 value %ld.", index, wNDimIdx,
                            weightNDimValue, index, baisNDim),
                    return ge::GRAPH_FAILED);
    }
    if (paramsInputInfo.numAntiquantOffset != 0) {
        // 检验weight的n轴和antiquantOffset的n轴一致
        auto antiquantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, index);
        OP_CHECK_NULL_WITH_CONTEXT(context, antiquantOffsetShape);
        int64_t antiquantOffsetNDim = antiquantOffsetShape->GetDim(0);
        OP_CHECK_IF(antiquantOffsetNDim != weightNDimValue,
                    OP_LOGE(context->GetNodeName(),
                            "weight[%lu] dim %lu value %ld should equal to antiquantOffset[%lu] dim 0 value %ld.",
                            index, wNDimIdx, weightNDimValue, index, antiquantOffsetNDim),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckCaseMultiSenario(const gert::InferShapeContext *context,
                                                                       const GMMAttrs &gmmAttrs,
                                                                       const GMMInputParamsInfo &paramsInputInfo) const
{
    const size_t &xSize = paramsInputInfo.numX;
    // check group size
    OP_CHECK_IF(CheckXWeightYGroupSizeMultiSenario(context, paramsInputInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The size of X,Y and weight are not equal."), return ge::GRAPH_FAILED);
    // check dimension
    OP_CHECK_IF(CheckDimNumNoSplit(context, paramsInputInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Dim num of tensor in tensor lists or grouplist is invalid."),
                return ge::GRAPH_FAILED);
    
    auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    size_t wKDimIdx = gmmAttrs.transposeWeight ? 1UL : 0UL;
    size_t wNDimIdx = gmmAttrs.transposeWeight ? 0UL : 1UL;

    int64_t weightKDimValue = wShape->GetDim(wKDimIdx);
    int64_t weightNDimValue = wShape->GetDim(wNDimIdx);

    for (size_t i = 0; i < xSize; i++) {
        xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, i);
        size_t xDimNum = xShape->GetDimNum();
        int64_t xKDimValue = xShape->GetDim(xDimNum - 1); // x always is not transposed

        wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
        weightKDimValue = wShape->GetDim(wKDimIdx);
        weightNDimValue = wShape->GetDim(wNDimIdx);
        // 校验M轴和batch轴大于等于0
        for (size_t n = 0; n < xDimNum - 1; n++) {
            int64_t xNDimValue = xShape->GetDim(n);
            OP_CHECK_IF(xNDimValue < 0,
                        OP_LOGE(context->GetNodeName(), "x[%lu] dim %lu value %ld should be more than or equal to 0.",
                                i, n, xNDimValue),
                        return ge::GRAPH_FAILED);
        }
        // 校验K轴和N轴大于0
        OP_CHECK_IF(
            xKDimValue <= 0,
            OP_LOGE(context->GetNodeName(),"x[%lu] dim %lu value %ld should more than 0.", i, xDimNum - 1, xKDimValue),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(weightNDimValue <= 0,
                    OP_LOGE(context->GetNodeName(), "w[%lu] dim %lu value %ld should more than 0.", i, wNDimIdx,
                            weightNDimValue),
                    return ge::GRAPH_FAILED);
        // 校验X和weight矩阵的K轴
        OP_CHECK_IF(xKDimValue != weightKDimValue,
                    OP_LOGE(context->GetNodeName(),
                            "x[%lu] dim %lu value %ld should equal to weight[%lu] dim 0 value %ld.", i, xDimNum - 1,
                            xKDimValue, i, weightKDimValue),
                    return ge::GRAPH_FAILED);
        // 校验 antiquantScale bias antiquantOffset 每个tensor的中n轴的大小
        OP_CHECK_IF(CheckTensorNDimMultiSenario(context, paramsInputInfo, wNDimIdx, weightNDimValue, i),
                    OP_LOGE(context->GetNodeName(), "CheckTensorNDimMultiSenario is failed."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckScenarioValidForShape(const gert::InferShapeContext *context,
                                                                            const GMMAttrs &gmmAttrs) const
{
    auto xSecondTensorShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 1);
    auto weightSecondTensorShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 1);
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    ge::DataType weightDtype = weightDesc->GetDataType();
    auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ge::DataType xDtype = xDesc->GetDataType();
    // 检验groupType
    OP_CHECK_IF((gmmAttrs.groupType != GMM_SPLIT_M) && (gmmAttrs.groupType != GMM_NO_SPLIT),
                OP_LOGE(context->GetNodeName(), "Invalid groupType, which can only be 0 or -1, but it is [%ld].",
                        gmmAttrs.groupType),
                return ge::GRAPH_FAILED);

    if (gmmAttrs.groupType == GMM_SPLIT_M) { // single/single/single Scenario
        OP_CHECK_IF(IsNonEmpty(xSecondTensorShape),
                    OP_LOGE(context->GetNodeName(), "The second tensor of tensor list x is not empty."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(IsNonEmpty(weightSecondTensorShape),
                    OP_LOGE(context->GetNodeName(), "The second tensor of tensor list weight is not empty."),
                    return ge::GRAPH_FAILED);

        // check split item value valid
        OP_CHECK_IF(gmmAttrs.splitItem != GMM_X_SEPARATED && gmmAttrs.splitItem != GMM_NO_SEPARATED,
                    OP_LOGE(context->GetNodeName(),
                            "Invalid splitItem, which can only be one of 2 or 3, but it is [%ld].", gmmAttrs.splitItem),
                    return ge::GRAPH_FAILED);
    } else { // multi/multi/multi Scenario
        // check split item value valid
        OP_CHECK_IF(!IsA16W8(xDtype, weightDtype),
                    OP_LOGE(context->GetNodeName(), "In multi/multi/multi Scenario, only support A16W8 format."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gmmAttrs.splitItem != GMM_X_Y_SEPARATED && gmmAttrs.splitItem != GMM_Y_SEPARATED,
                    OP_LOGE(context->GetNodeName(),
                            "Invalid splitItem, which can only be one of 0 or 1, but it is [%ld].", gmmAttrs.splitItem),
                    return ge::GRAPH_FAILED);
    }
    // check activation is null
    OP_CHECK_IF(gmmAttrs.activeType != static_cast<int64_t>(GMMActType::GMM_ACT_TYPE_NONE),
                OP_LOGE(context->GetNodeName(), "Activation function is not supported in weight quant mode now."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeForGrouplist(const gert::InferShapeContext *context,
                                                                        const gert::Shape *groupListShape) const
{
    OP_CHECK_IF(groupListShape->GetDimNum() != 1,
                OP_LOGE(context->GetNodeName(),
                        "In single-single-single scenario, "
                        "the groupList only support 1 dim num for now, but the actual dim num is [%zu].",
                        groupListShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupListShape->GetDim(0) <= 0,
                OP_LOGE(context->GetNodeName(),
                        "The groupList 1st dim value should be greater than 0, but the actual value is [%ld].",
                        groupListShape->GetDim(0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupListShape->GetDim(0) > GMM_MAX_GROUP_LIST_SIZE_TENSOR,
                OP_LOGE(context->GetNodeName(),
                        "Only support [%ld] groups at MAX for now, but the actual group number is [%ld].",
                        GMM_MAX_GROUP_LIST_SIZE_TENSOR, groupListShape->GetDim(0)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckGroupAntiS(const gert::Shape *tensorShape,
                                                                 const gert::InferShapeContext *context,
                                                                 const std::string &tensorType) const
{
    size_t tensorDimNum = tensorShape->GetDimNum();
    // antiquantscale的Shape为(g, k/groupSize, n)/(g, n, k/groupsize),维度数为3,单独校验
    OP_CHECK_IF(tensorDimNum != 3,
                OP_LOGE(context->GetNodeName(),
                        "When %s is not null, its dim should be 3, but the actual dim num is [%zu].",
                        tensorType.c_str(), tensorDimNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tensorShape->GetDim(0) != groupNum_,
                OP_LOGE(context->GetNodeName(),
                        "The first dim of %s should be g, which is %ld, but the actual shape is %ld.",
                        tensorType.c_str(), groupNum_, tensorShape->GetDim(0)),
                return ge::GRAPH_FAILED);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "Attrs is nullptr."), return ge::GRAPH_FAILED);
    const bool *transposeWPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_W);
    OP_CHECK_NULL_WITH_CONTEXT(context, transposeWPtr);
    auto antiSN = *transposeWPtr ? tensorShape->GetDim(1) : tensorShape->GetDim(2);
    OP_CHECK_IF(antiSN != weightNDim_,
                OP_LOGE(context->GetNodeName(),
                        "The n dim of %s should be weight's n, which is %ld, but the actual shape is %ld.",
                        tensorType.c_str(), weightNDim_, antiSN),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckPertokenScaleForA8W4(const gert::Shape *tensorShape,
                                                                           const gert::InferShapeContext *context,
                                                                           const std::string &tensorType) const
{
    size_t tensorDimNum = tensorShape->GetDimNum();
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    ge::DataType weightDtype = weightDesc->GetDataType();
    auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    ge::DataType xDtype = xDesc->GetDataType();
    size_t tensorDimNumExp = IsS8S4NZ(xDtype, weightDtype) ? 1 : 2; // S8S4维度为1，MxA8W4维度为2
    OP_CHECK_IF(tensorDimNum != tensorDimNumExp,
                OP_LOGE(context->GetNodeName(),
                        "When %s is not null, its dim should be [%zu], but the actual dim num is [%zu].",
                        tensorType.c_str(), tensorDimNumExp, tensorDimNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tensorShape->GetDim(0) != xMDim_,
                OP_LOGE(context->GetNodeName(),
                        "The shape of %s should be (m), which is (%ld), but the actual shape is (%ld).",
                        tensorType.c_str(), xMDim_, tensorShape->GetDim(0)),
                return ge::GRAPH_FAILED);
    if (IsMxA8W4NZ(xDtype, weightDtype)) {
        OP_CHECK_IF(tensorShape->GetDim(1) != weightKDim_ / 32, // 32含义：groupsize大小
                    OP_LOGE(context->GetNodeName(),
                            "The shape of %s should be (m, k/32), which is (%ld, %ld), but the actual "
                            "shape is (%ld, %ld).",
                            tensorType.c_str(), xMDim_, weightKDim_ / 32, tensorShape->GetDim(0),
                            tensorShape->GetDim(1)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeForTensorList(const gert::InferShapeContext *context,
                                                                         size_t gmm_index,
                                                                         const std::string &tensorType) const
{
    auto tensorShape = context->GetDynamicInputShape(gmm_index, 0);
    if (IsNonEmpty(tensorShape)) {
        size_t tensorDimNum = tensorShape->GetDimNum();
        auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
        ge::DataType weightDtype = weightDesc->GetDataType();
        auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
        ge::DataType xDtype = xDesc->GetDataType();
        // check 3 dim antiquantscale
        if (gmm_index == GMM_INDEX_IN_ANTIQUANT_SCALE &&
            (IsA16MxFp4NZ(xDtype, weightDtype) || IsA8W4(xDtype, weightDtype))) {
            OP_CHECK_IF(CheckGroupAntiS(tensorShape, context, tensorType) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "Check GroupAntiquanScale failed."), return ge::GRAPH_FAILED);
        } else {
            if (IsA8W4(xDtype, weightDtype) && (gmm_index == GMM_INDEX_IN_PERTOKEN_SCALE)) {
                OP_CHECK_IF(CheckPertokenScaleForA8W4(tensorShape, context, tensorType) != ge::GRAPH_SUCCESS,
                            OP_LOGE(context->GetNodeName(), "CheckPertokenScaleForA8W4 failed."),
                            return ge::GRAPH_FAILED);
            } else {
                // check the dim of antiquantscale、 antiquantoffset、bias、scale
                OP_CHECK_IF(tensorDimNum != 2,
                            OP_LOGE(context->GetNodeName(),
                                    "When %s is not null, its dim should be 2, but the actual dim num is [%zu].",
                                    tensorType.c_str(), tensorDimNum),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(
                    tensorShape->GetDim(0) != groupNum_ || tensorShape->GetDim(1) != weightNDim_,
                    OP_LOGE(
                        context->GetNodeName(),
                        "The shape of %s should be (g, n), which is (%ld, %ld), but the actual shape is (%ld, %ld).",
                        tensorType.c_str(), groupNum_, weightNDim_, tensorShape->GetDim(0), tensorShape->GetDim(1)),
                    return ge::GRAPH_FAILED);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeForWeightQuantParam(const gert::InferShapeContext *context) const
{
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    auto offsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_OFFSET, 0);
    auto perTokenScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    auto antiquantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, 0);
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    ge::DataType weightDtype = weightDesc->GetDataType();
    auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto xDtype = xDesc->GetDataType();
    OP_CHECK_IF(IsNonEmpty(antiquantOffsetShape) && (FP8_SUPPORT_SET.find(weightDtype) != FP8_SUPPORT_SET.end() ||
                                                     weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT),
                OP_LOGE(context->GetNodeName(),
                    "In weight quant case, only support antiquantOffset is none when weightDtype is fp8/hif8/fp4."), 
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsNonEmpty(scaleShape) && !IsS8S4NZ(xDtype, weightDtype),
                OP_LOGE(context->GetNodeName(),
                        "In weight quant case, scale must be empty when xDtype-weightDtype is not int8-int4."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsNonEmpty(scaleShape) && IsS8S4NZ(xDtype, weightDtype),
                OP_LOGE(context->GetNodeName(),
                    "In weight quant case, scale must not be empty when xDtype-weightDtype is int8-int4."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsNonEmpty(offsetShape),
                OP_LOGE(context->GetNodeName(), "In weight quant case, offset must be empty."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsNonEmpty(perTokenScaleShape) && (!IsMxA8W4NZ(xDtype, weightDtype) && !IsS8S4NZ(xDtype, weightDtype)),
                OP_LOGE(context->GetNodeName(), "In weight quant case, pertokenscale must be empty when "
                                                "xDtype-weightDtype is not int8-int4 or float8_e4m3fn-float4_e2m1."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsNonEmpty(perTokenScaleShape) && (IsMxA8W4NZ(xDtype, weightDtype) || IsS8S4NZ(xDtype, weightDtype)),
                OP_LOGE(context->GetNodeName(), "In weight quant case, pertokenscale must not be empty when "
                                                "xDtype-weightDtype is int8-int4 or float8_e4m3fn-float4_e2m1."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsNonEmpty(antiquantScaleShape), OP_LOGE(context->GetNodeName(), 
            "In weight quant case, antiquantScale must not be empty."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeForTensorList(context, GMM_INDEX_IN_ANTIQUANT_SCALE, "antiquantScale") != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShapeForAntiquantScale failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeForTensorList(context, GMM_INDEX_IN_ANTIQUANT_OFFSET, "antiquantOffset") != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShapeForAntiquantOffset failed."), return ge::GRAPH_FAILED);
    if (IsMxA8W4NZ(xDtype, weightDtype) || IsS8S4NZ(xDtype, weightDtype)) {
        OP_CHECK_IF(CheckShapeForTensorList(context, GMM_INDEX_IN_PERTOKEN_SCALE, "pertokenScale") !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForpertokenScale failed."), return ge::GRAPH_FAILED);}
    if(IsS8S4NZ(xDtype, weightDtype)){
        OP_CHECK_IF(CheckShapeForTensorList(context, GMM_INDEX_IN_SCALE, "scale") != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForscale failed."), return ge::GRAPH_FAILED);}
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckGroupSize(const gert::InferShapeContext *context,
                                                                const GMMAttrs &gmmAttrs) const
{
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    auto weightDtype = weightDesc->GetDataType();
    auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto xDtype = xDesc->GetDataType();
    int64_t groupSize = 0;
    // 3含义，当前shape为(g,k/groupsize,n)或者(g,n,k/groupSize), 在伪量化Mx场景出现
    if (IsA16MxFp4NZ(xDtype, weightDtype) || IsMxA8W4NZ(xDtype, weightDtype) || IsS8S4NZ(xDtype, weightDtype)) {
        auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
        auto antiquantScaleDimNum = antiquantScaleShape->GetDimNum();
        // 2含义: (g,k/groupSize,n)的k轴索引,此处groupNum是K轴上量化分组的groupNum，与groupNum_含义不同
        int64_t groupNum = gmmAttrs.transposeWeight ? antiquantScaleShape->GetDim(antiquantScaleDimNum - 1) :
                                                      antiquantScaleShape->GetDim(antiquantScaleDimNum - 2);
        OP_CHECK_IF(groupNum <= 0, OP_LOGE(context->GetNodeName(), "GroupNum must be greater than 0."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(weightKDim_ % groupNum != 0,
                    OP_LOGE(context->GetNodeName(), "GroupNum must be multiple of the k axis of weight."),
                    return ge::GRAPH_FAILED);
        groupSize = weightKDim_ / groupNum;
    }
    if (IsS8S4NZ(xDtype, weightDtype)) {
        // 伪量化S8S4场景支持groupsize为128/256/512
        OP_CHECK_IF(groupSize != 128 && groupSize != 256 && groupSize != 512,
                    OP_LOGE(context->GetNodeName(), "groupSize must be 128/256/512, but current groupSize is (%ld).",
                            groupSize),
                    return ge::GRAPH_FAILED);
    } else {
        // 当前伪量化非S8S4仅支持groupSize为0或为32的整数倍
        OP_CHECK_IF(groupSize != 32 && groupSize != 0,
                    OP_LOGE(context->GetNodeName(),
                            "groupSize must be a multiple of 32, but current groupSize is (%ld).", groupSize),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::GetNumOfInputs(const gert::InferShapeContext *context,
                                                                GMMInputParamsInfo &paramsInputInfo) const
{
    ge::graphStatus res = ge::GRAPH_SUCCESS;
    const gert::Shape *shape = nullptr;
    struct ParamInfoTmp {
        int index;
        size_t &count;
        const char *name;
    };
    ParamInfoTmp params[] = {{GMM_INDEX_IN_X, paramsInputInfo.numX, "numX"},
                             {GMM_INDEX_IN_WEIGHT, paramsInputInfo.numWeight, "numWeight"},
                             {GMM_INDEX_IN_BIAS, paramsInputInfo.numBias, "numBias"},
                             {GMM_INDEX_IN_ANTIQUANT_SCALE, paramsInputInfo.numAntiquantScale, "numAntiquantScale"},
                             {GMM_INDEX_IN_ANTIQUANT_OFFSET, paramsInputInfo.numAntiquantOffset, "numAntiquantOffset"}};

    for (auto &param : params) {
        param.count = 0;
        for (int i = 0; i < GMM_MAX_GROUP_LIST_SIZE_ARRAY ; i++) {
            shape = context->GetDynamicInputShape(param.index, param.count);
            if (!IsNonEmpty(shape)) {
                break;
            }
            ++param.count;
        }
        OP_CHECK_IF(param.count >= GMM_MAX_GROUP_LIST_SIZE_ARRAY,
                        OP_LOGE(context->GetNodeName(),
                                "In multi/multi/multi Scenario, each tensorlist's length cannot exceed 128"),
                        return ge::GRAPH_FAILED);
        OP_LOGI(context->GetNodeName(), "%s = %lu", param.name, param.count);
    }
    return res;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeForWeightQuantParamMultiScenario(
    const gert::InferShapeContext *context, const GMMInputParamsInfo &paramsInputInfo) const
{
    // 检测 bias antiquantScale antiquantOffset的 tensorListsize 需要等于 weightSize
    auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, 0);
    if (IsNonEmpty(biasShape)) {
        OP_CHECK_IF(paramsInputInfo.numBias != paramsInputInfo.numWeight,
                    OP_LOGE(context->GetNodeName(),
                            "Bais size should be equal to weight size, actual size are [%lu] and [%lu]",
                            paramsInputInfo.numBias, paramsInputInfo.numWeight),
                    return ge::GRAPH_FAILED);
    }

    auto antiquantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, 0);
    if (IsNonEmpty(antiquantOffsetShape)) {
        OP_CHECK_IF(paramsInputInfo.numAntiquantOffset != paramsInputInfo.numWeight,
                    OP_LOGE(context->GetNodeName(),
                            "AntiquantOffset size should be equal to weight size, actual size are [%lu] and [%lu]",
                            paramsInputInfo.numAntiquantOffset, paramsInputInfo.numWeight),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(paramsInputInfo.numAntiquantScale != paramsInputInfo.numWeight,
                OP_LOGE(context->GetNodeName(),
                        "AntiquantScale size should be equal to weight size, actual size are [%lu] and [%lu]",
                        paramsInputInfo.numAntiquantScale, paramsInputInfo.numWeight),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeValid(const gert::InferShapeContext *context,
                                                                 const GMMAttrs &gmmAttrs)
{
    if (gmmAttrs.groupType == GMM_NO_SPLIT) {
        GMMInputParamsInfo paramsInputInfo{0, 0, 0, 0, 0, 0, 0};
        OP_CHECK_IF(GetNumOfInputs(context, paramsInputInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "GetNumOfInputs failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckShapeForWeightQuantParamMultiScenario(context, paramsInputInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForWeightQuantParamMultiScenario failed."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckCaseMultiSenario(context, gmmAttrs, paramsInputInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckCaseMultiSenario failed."), return ge::GRAPH_FAILED);
    } else {
        auto groupListShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
        OP_CHECK_NULL_WITH_CONTEXT(context, groupListShape);
        OP_CHECK_IF(CheckShapeForGrouplist(context, groupListShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForGrouplist failed."), return ge::GRAPH_FAILED);
        groupNum_ = groupListShape->GetDim(0);
        OP_CHECK_IF(CheckShapeForXAndWeight(context, gmmAttrs) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForXAndWeight failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckShapeForTensorList(context, GMM_INDEX_IN_BIAS, "bias") != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForBias failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckShapeForWeightQuantParam(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForWeightQuantParam failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckGroupSize(context, gmmAttrs) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckGroupSize failed."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool IsUnknownShape(const gert::Shape *shape)
{
    if (IsNonEmpty(shape)) {
        size_t size = shape->GetDimNum();
        for (size_t i = 0; i < size; i++) {
            if (shape->GetDim(i) == UNKNOWN_SHAPE_VALUE || shape->GetDim(i) == SHAPE_UNKNOWN_DIM_NUM) {
                return true;
            }
        }
        return false;
    }
    return false;
}

bool CheckUnknownShape(const gert::InferShapeContext *context)
{
    bool hasUnknownShape = false;
    auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    auto antiquantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, 0);
    auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, 0);
    auto groupListShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
    hasUnknownShape = IsUnknownShape(xShape) || IsUnknownShape(weightShape) || IsUnknownShape(antiquantScaleShape) ||
                      IsUnknownShape(groupListShape);
    if (!hasUnknownShape && IsNonEmpty(antiquantOffsetShape)) {
        hasUnknownShape |= IsUnknownShape(antiquantOffsetShape);
    }
    if (!hasUnknownShape && IsNonEmpty(biasShape)) {
        hasUnknownShape |= IsUnknownShape(biasShape);
    }
    return hasUnknownShape;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShape(const gert::InferShapeContext *context,
                                                            const GroupedMatmulCommonUtil &commonUtil)
{
    if (xMDim_ < 0 || CheckUnknownShape(context)) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(CheckScenarioValidForShape(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckScenarioValidForShape failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeValid(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShapeValid failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::UpdateShapeY(gert::InferShapeContext *context, size_t idxY,
                                                              std::vector<int64_t> &yDims) const
{
    gert::Shape *yShape = context->GetOutputShape(idxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(yDims.size());
    for (size_t dim = 0; dim < yDims.size(); ++dim) {
        yShape->SetDim(dim, yDims[dim]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::UpdateShapeYMultiDim(gert::InferShapeContext *context, size_t idxY,
                                                                      const gert::Shape *xShape,
                                                                      const gert::Shape *weightShape) const
{
    gert::Shape *yShape = context->GetOutputShape(idxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *xShape;
    size_t dimY = yShape->GetDimNum();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *transposeWPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_W);
    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_X);

    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    if (transposeWPtr != nullptr && *transposeWPtr) {
        yShape->SetDim(dimY - 1, weightShape->GetDim(weightShape->GetDimNum() - 2)); // -2: transpose weight
    } else {
        yShape->SetDim(dimY - 1, weightShape->GetDim(weightShape->GetDimNum() - 1));
    }
    if (transposeXPtr != nullptr && *transposeXPtr) {
        yShape->SetDim(dimY - 2, xShape->GetDim(xShape->GetDimNum() - 1)); // -2: last two dim of Y
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::InferOutShape(gert::InferShapeContext *context,
                                                               const GMMAttrs &gmmAttrs) const
{
    if (gmmAttrs.groupType == GMM_NO_SPLIT) {
        size_t idx = 0;
        size_t idw = 0;
        const gert::Shape *w0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, w0Shape);
        for (int i = 0; i < GMM_MAX_GROUP_LIST_SIZE_ARRAY; i++) {
            const gert::Shape *xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, idx);
            if (xShape == nullptr) {
                break;
            }
            ++idx;
            const gert::Shape *wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, idw);
            if (wShape) {
                ++idw;
            } else {
                wShape = w0Shape;
            }
            OP_CHECK_IF(UpdateShapeYMultiDim(context, GMM_INDEX_OUT_Y + idx - 1, xShape, wShape) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return ge::GRAPH_FAILED);
        }
    } else {
        std::vector<int64_t> yDims = {xMDim_, weightNDim_};
        OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y, yDims) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Failed to update y shape."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckScaleDtypeForS8S4(const gert::InferDataTypeContext *context) const
{
    auto perTokenScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    auto scaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_SCALE, 0);
    auto antiquantScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    OP_CHECK_IF(scaleDtype != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "scaleDtype datatype [%s] does not match float32.",
                        ge::TypeUtils::DataTypeToAscendString(scaleDtype).GetString()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(antiquantScaleDtype != ge::DT_FLOAT16,
                OP_LOGE(context->GetNodeName(), "antiquantScaleDtype datatype [%s] does not match float16.",
                        ge::TypeUtils::DataTypeToAscendString(antiquantScaleDtype).GetString()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(perTokenScaleDtype != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "perTokenScaleDtype datatype [%s] does not match float32.",
                        ge::TypeUtils::DataTypeToAscendString(perTokenScaleDtype).GetString()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckBiasDtype(const gert::InferDataTypeContext *context) const
{
    auto xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto biasDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_BIAS, 0);
    OP_CHECK_IF(BIAS_TYPE_SUPPORT_MAP.find(xDtype) == BIAS_TYPE_SUPPORT_MAP.end(),
              OP_LOGE(context->GetNodeName(), "Cannot find bias dtype match with xDtype [%s].",
                        ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),return ge::GRAPH_FAILED);
    OP_CHECK_IF(BIAS_TYPE_SUPPORT_MAP.at(xDtype).find(biasDtype) == BIAS_TYPE_SUPPORT_MAP.at(xDtype).end(),
              OP_LOGE(context->GetNodeName(), "Data type [%s] is not supported for bias, when xDtype is [%s].",
                        ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString(),
                        ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckTensorListDataType(const gert::InferDataTypeContext *context,
                                                                         uint32_t index, const ge::DataType dtype) const
{
    size_t inIdx = 0;
    for (int i = 0; i < GMM_MAX_GROUP_LIST_SIZE_ARRAY; i++)  {
        auto iDtype = context->GetDynamicInputDataType(index, inIdx);
        if (iDtype == ge::DT_UNDEFINED) {
            break;
        }
        OP_CHECK_IF(iDtype != dtype,
                    OP_LOGE(context->GetNodeName(), "data type of tensors in a tensorList should all be the same!"),
                    return ge::GRAPH_FAILED);
        ++inIdx;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckMatmulDataType(const gert::InferDataTypeContext *context,
                                                                     const ge::DataType xDtype,
                                                                     const ge::DataType weightDtype,
                                                                     const ge::DataType biasDtype,
                                                                     const ge::DataType antiquantScaleDtype) const
{
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_X, xDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "x dtype does not match with required dtype[%s].",
                        ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_WEIGHT, weightDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "weight dtype does not match with required dtype[%s].",
                        ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_BIAS, biasDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "bias dtype does not match with required dtype[%s].",
                        ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_ANTIQUANT_SCALE, antiquantScaleDtype) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "antiquantScaleDtype dtype does not match with required dtype[%s].",
                        ge::TypeUtils::DataTypeToAscendString(antiquantScaleDtype).GetString()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckDtype(const gert::InferDataTypeContext *context) const
{
    auto xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto weightDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_WEIGHT, 0);
    // mandory param dtype check
    OP_CHECK_IF(X_TYPE_SUPPORT_SET.find(xDtype) == X_TYPE_SUPPORT_SET.end(),
                OP_LOGE(context->GetNodeName(), "Data type [%s] is not supported for x's 1st tensor.",
                        ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),  return ge::GRAPH_FAILED);
    OP_CHECK_IF(WEIGHT_TYPE_SUPPORT_SET.find(weightDtype) == WEIGHT_TYPE_SUPPORT_SET.end(),
                OP_LOGE(context->GetNodeName(), "Data type [%s] is not supported for weight.",
                        ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()), return ge::GRAPH_FAILED);
    auto antiquantScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    auto perTokenScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    if (weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT) {
        OP_CHECK_IF(perTokenScaleDtype != ge::DT_FLOAT8_E8M0 && IsMxA8W4NZ(xDtype, weightDtype),
                    OP_LOGE(context->GetNodeName(),"Only support float8_e8m0 for perTokenScaleDtype when mxA8W4, "
                            "but got [%s].", ge::TypeUtils::DataTypeToAscendString(perTokenScaleDtype).GetString()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(antiquantScaleDtype != ge::DT_FLOAT8_E8M0, OP_LOGE(context->GetNodeName(),
                    "Only support float8_e8m0 for antiquantScaleDataType when weight is fp4, but got [%s].",
                    ge::TypeUtils::DataTypeToAscendString(antiquantScaleDtype).GetString()),
            return ge::GRAPH_FAILED);
    } else if (IsS8S4NZ(xDtype, weightDtype)) {
        OP_CHECK_IF(CheckScaleDtypeForS8S4(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckScaleDtypeForS8S4 failed."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(antiquantScaleDtype != xDtype,
                    OP_LOGE(context->GetNodeName(), "AntiquantScale datatype [%s] does not match xDtype [%s].",
                            ge::TypeUtils::DataTypeToAscendString(antiquantScaleDtype).GetString(),
                            ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()), return ge::GRAPH_FAILED);
    }
    if (FP8_SUPPORT_SET.find(weightDtype) == FP8_SUPPORT_SET.end() && weightDtype != ge::DT_FLOAT4_E2M1 &&
        weightDtype != ge::DT_FLOAT && weightDtype != ge::DT_INT4 && weightDtype != ge::DT_INT32) {
        auto antiquantOffsetDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_ANTIQUANT_OFFSET, 0);
        OP_CHECK_IF(antiquantOffsetDtype != xDtype,
                    OP_LOGE(context->GetNodeName(), "AntiquantOffset datatype [%s] does not match xDtype [%s].",
                            ge::TypeUtils::DataTypeToAscendString(antiquantOffsetDtype).GetString(),
                            ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            CheckTensorListDataType(context, GMM_INDEX_IN_ANTIQUANT_OFFSET, antiquantOffsetDtype) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "antiquantOffsetDtype dtype does not match with required dtype[%s].",
                    ge::TypeUtils::DataTypeToAscendString(antiquantOffsetDtype).GetString()), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(CheckBiasDtype(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckBiasDtype failed."), return ge::GRAPH_FAILED);
    auto biasDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_BIAS, 0);
    OP_CHECK_IF(CheckMatmulDataType(context, xDtype, weightDtype, biasDtype, antiquantScaleDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckMatmulDataType is failed!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::InferOutDtype(gert::InferDataTypeContext *context) const
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *outputDtype = attrs->GetInt(GMM_INDEX_ATTR_OUTPUT_DTYPE);
    ge::DataType yDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto it = GMM_OUTPUT_DTYPE_MAP.find(*outputDtype);
    OP_CHECK_IF(it == GMM_OUTPUT_DTYPE_MAP.end(),
                OP_LOGE(context->GetNodeName(),
                        "The output dtype should be bfloat16 or float16 , but the actual output dtype is [%s].",
                        ge::TypeUtils::DataTypeToAscendString(yDtype).GetString()),
                return ge::GRAPH_FAILED);
    yDtype = it->second;
    context->SetOutputDataType(GMM_INDEX_OUT_Y, yDtype);
    return ge::GRAPH_SUCCESS;
}

} // namespace ops