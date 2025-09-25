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
static const std::unordered_set<ge::DataType> X_TYPE_SUPPORT_SET = {ge::DT_FLOAT16, ge::DT_BF16};
static const std::unordered_set<ge::DataType> WEIGHT_TYPE_SUPPORT_SET = {
    ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT};
static const std::map<ge::DataType, std::unordered_set<ge::DataType>> BIAS_TYPE_SUPPORT_MAP = {
    {ge::DT_FLOAT16, {ge::DT_FLOAT16}}, {ge::DT_BF16, {ge::DT_BF16, ge::DT_FLOAT}}};
static const std::unordered_set<ge::DataType> FP8_SUPPORT_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8};
const int64_t UNKNOWN_SHAPE_VALUE = -1;
const int64_t SHAPE_UNKNOWN_DIM_NUM = -2;
const int64_t B4_NUMS_IN_B32 = 8;

static bool inline IsNonEmpty(const gert::Shape *shape)
{
    return (shape != nullptr && !(shape->GetDimNum() == 1 && shape->GetDim(0) == 0));
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
    weightKDim_ = gmmAttrs.transposeWeight ? weightShape->GetDim(weightdimNum_ - 1)
                                           : weightShape->GetDim(weightdimNum_ - PENULTIMATE_DIM);
    weightNDim_ = gmmAttrs.transposeWeight ? weightShape->GetDim(weightdimNum_ - PENULTIMATE_DIM)
                                           : weightShape->GetDim(weightdimNum_ - 1);
    if(weightDtype == ge::DT_FLOAT){
        weightNDim_ = weightNDim_ * B4_NUMS_IN_B32;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeForXAndWeight(const gert::InferShapeContext *context,
                                                                     const GMMAttrs &gmmAttrs) const
{
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    ge::DataType weightDtype = weightDesc->GetDataType();
    OP_CHECK_IF(xKDim_ != weightKDim_, OP_LOGE(context->GetNodeName(),
                  "The k dim of x should be equal to the k dim of weight, but x's k is [%ld] and weight's k is [%ld].",
                  xKDim_, weightKDim_), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightNDim_ > 0 && weightNDim_ <= GMM_MAX_INNER_AXIS), OP_LOGE(context->GetNodeName(),
                    "The n dim value should be positive and not larger than [%ld], but the actual value is [%ld].",
                    GMM_MAX_INNER_AXIS, weightNDim_),return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightKDim_ > 0 && weightKDim_ <= GMM_MAX_INNER_AXIS), OP_LOGE(context->GetNodeName(),
                        "The k dim value should be positive and not larger than [%ld], but the actual value is [%ld].",
                        GMM_MAX_INNER_AXIS, weightKDim_), return ge::GRAPH_FAILED);
    if (weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT) {
        OP_CHECK_IF(
            !((weightNDim_ % GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0) &&
              (weightKDim_ % GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0)),
            OP_LOGE(
                context->GetNodeName(),
                "The value of dim n, k should be an integer multiple of [%ld], but actual n is [%ld], k is [%ld].",
                GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT, weightNDim_, weightKDim_), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            !((weightNDim_ % GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT == 0) &&
              (weightKDim_ % GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT == 0)),
            OP_LOGE(
                context->GetNodeName(),
                "The value of dim n, k should be an integer multiple of [%ld], but actual n is [%ld], k is [%ld].",
                GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT, weightNDim_, weightKDim_), return ge::GRAPH_FAILED);
    }
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        OP_CHECK_IF(xdimNum_ != GMM_MIN_FM_DIM || weightdimNum_ != GMM_SPLIT_M_SINGLE_WEIGHT_DIM,
                  OP_LOGE(context->GetNodeName(),
                            "When split m, x dim num should be 2, weight dim num should be 3, "
                            "but the actual x dim num is [%zu], actual weight dim num is [%zu].",
                            xdimNum_, weightdimNum_), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            weightShape->GetDim(0) != groupNum_,
            OP_LOGE(
                context->GetNodeName(),
                "When split m, 1st dim value of weight should be g, which is [%ld], but the actual value is [%ld].",
                groupNum_, weightShape->GetDim(0)), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckScenarioValidForShape(const gert::InferShapeContext *context,
                                                                        const GMMAttrs &gmmAttrs) const
{
    // only support for single/single/single Scenario
    auto xSecondTensorShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 1);
    auto weightSecondTensorShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 1);
    OP_CHECK_IF(
        IsNonEmpty(xSecondTensorShape),
        OP_LOGE(
            context->GetNodeName(),
            "Only support single/single/single scenario for now, but the second tensor of tensor list x is not empty."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsNonEmpty(weightSecondTensorShape),
              OP_LOGE(context->GetNodeName(),
                        "Only support single/single/single scenario for now, but the second tensor of tensor list "
                        "weight is not empty."),
              return ge::GRAPH_FAILED);

    // check split item value valid
    OP_CHECK_IF(gmmAttrs.splitItem != GMM_X_SEPARATED && gmmAttrs.splitItem != GMM_NO_SEPARATED,
              OP_LOGE(context->GetNodeName(), "Invalid splitItem, which can only be one of 2 or 3, but it is [%ld].",
                        gmmAttrs.splitItem),
              return ge::GRAPH_FAILED);
    OP_CHECK_IF(gmmAttrs.groupType != GMM_SPLIT_M,
              OP_LOGE(context->GetNodeName(), "Invalid groupType, which can only be 0, but it is [%ld].",
                        gmmAttrs.groupType),
              return ge::GRAPH_FAILED);
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
                        "The groupList only support 1 dim num for now, but the actual dim num is [%zu].",
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

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeForTensorList(const gert::InferShapeContext *context,
                                                                     size_t gmm_index,
                                                                     const std::string &tensorType) const
{
    auto tensorShape = context->GetDynamicInputShape(gmm_index, 0);
    if (IsNonEmpty(tensorShape)) {
        size_t tensorDimNum = tensorShape->GetDimNum();
        auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
        ge::DataType weightDtype = weightDesc->GetDataType();
        if (gmm_index == GMM_INDEX_IN_ANTIQUANT_SCALE &&
            (weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT)) {
            // antiquantscale的Shape为(g, K/groupSize, n),维度数为3,单独校验
            OP_CHECK_IF(tensorDimNum != 3,
                      OP_LOGE(context->GetNodeName(),
                                "When %s is not null, its dim should be 3, but the actual dim num is [%zu].",
                                tensorType.c_str(), tensorDimNum),
                      return ge::GRAPH_FAILED);
            OP_CHECK_IF(tensorShape->GetDim(0) != groupNum_ || tensorShape->GetDim(2) != weightNDim_,
                      OP_LOGE(context->GetNodeName(),
                                "The last dim of %s should be n, which is %ld, but the actual shape is (%ld, %ld).",
                                tensorType.c_str(), groupNum_, weightNDim_, tensorShape->GetDim(2)),
                      return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(tensorDimNum != 2,
                      OP_LOGE(context->GetNodeName(),
                                "When %s is not null, its dim should be 2, but the actual dim num is [%zu].",
                                tensorType.c_str(), tensorDimNum),
                      return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                tensorShape->GetDim(0) != groupNum_ || tensorShape->GetDim(1) != weightNDim_,
                OP_LOGE(context->GetNodeName(),
                          "The shape of %s should be (g, n), which is (%ld, %ld), but the actual shape is (%ld, %ld).",
                          tensorType.c_str(), groupNum_, weightNDim_, tensorShape->GetDim(0), tensorShape->GetDim(1)),
                return ge::GRAPH_FAILED);
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
    OP_CHECK_IF(IsNonEmpty(antiquantOffsetShape) && (FP8_SUPPORT_SET.find(weightDtype) != FP8_SUPPORT_SET.end() ||
                                                   weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT),
              OP_LOGE(context->GetNodeName(),
                        "In weight quant case, only support antiquantOffset is none when weightDtype is fp8/hif8/fp4."),
              return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        IsNonEmpty(scaleShape) || IsNonEmpty(offsetShape) || IsNonEmpty(perTokenScaleShape),
        OP_LOGE(context->GetNodeName(), "In weight quant case, scale, offset, and pertokenScale must be empty."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!IsNonEmpty(antiquantScaleShape),
              OP_LOGE(context->GetNodeName(), "In weight quant case, antiquantScale must be not empty."),
              return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeForTensorList(context, GMM_INDEX_IN_ANTIQUANT_SCALE, "antiquantScale") != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckShapeForAntiquantScale failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeForTensorList(context, GMM_INDEX_IN_ANTIQUANT_OFFSET, "antiquantOffset") != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckShapeForAntiquantOffset failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckGroupSize(const gert::InferShapeContext *context,
                                                            const GMMAttrs &gmmAttrs) const {
    auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    auto antiquantScaleDimNum = antiquantScaleShape->GetDimNum();
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    int64_t groupSize = 0;
    // 3含义，当前shape为(g,k/groupsize,n)或者(g,n,k/groupSize), 在伪量化Mx场景出现
    if (antiquantScaleDimNum == 3) {
        // 2含义: (g,k/groupSize,n)的k轴索引,此处groupNum是K轴上量化分组的groupNum，与groupNum_含义不同
        int64_t groupNum = gmmAttrs.transposeWeight ? antiquantScaleShape->GetDim(antiquantScaleDimNum - 1)
                                                    : antiquantScaleShape->GetDim(antiquantScaleDimNum - 2);
        OP_CHECK_IF(groupNum <= 0, OP_LOGE(context->GetNodeName(), "GroupNum must be greater than 0."),
                  return ge::GRAPH_FAILED);
        groupSize = weightKDim_ / groupNum;
    }
    // 当前伪量化仅支持groupSize为0或为32的整数倍
    OP_CHECK_IF(groupSize % 32 != 0, OP_LOGE(context->GetNodeName(), "groupSize must be a multiple of 32."),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeValid(const gert::InferShapeContext *context,
                                                             const GMMAttrs &gmmAttrs)
{
    auto groupListShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListShape);
    OP_CHECK_IF(CheckShapeForGrouplist(context, groupListShape) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckShapeForGrouplist failed."), return ge::GRAPH_FAILED);
    groupNum_ = groupListShape->GetDim(0);
    OP_CHECK_IF(CheckGroupSize(context, gmmAttrs) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckGroupSize failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeForXAndWeight(context, gmmAttrs) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckShapeForXAndWeight failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeForTensorList(context, GMM_INDEX_IN_BIAS, "bias") != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckShapeForBias failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeForWeightQuantParam(context) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckShapeForWeightQuantParam failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckFormatValid(const gert::InferShapeContext *context) const
{
    const auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const auto xFormat = xDesc->GetOriginFormat();
    OP_CHECK_IF(xFormat == ge::FORMAT_FRACTAL_NZ, OP_LOGE(context->GetNodeName(), "Format of x does not support NZ."),
              return ge::GRAPH_FAILED);

    const auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    const auto weightFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(weightDesc->GetStorageFormat()));
    const auto weightDtype = weightDesc->GetDataType();
    if (weightDtype != ge::DT_FLOAT4_E2M1 && weightDtype != ge::DT_FLOAT) {
        OP_CHECK_IF(
            weightFormat != ge::FORMAT_ND && weightFormat != ge::FORMAT_NCL,
            OP_LOGE(context->GetNodeName(), "Format of weight only support ND or NCL, but given format is [%s].",
                      ge::TypeUtils::FormatToAscendString(weightFormat).GetString()),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(weightFormat != ge::FORMAT_FRACTAL_NZ,
                  OP_LOGE(context->GetNodeName(),
                            "Format of weight only support NZ when weightDtype is fp4, but given format is [%s].",
                            ge::TypeUtils::FormatToAscendString(weightFormat).GetString()),
                  return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

bool IsUnknownShape(const gert::Shape *shape) {
    size_t size = shape->GetDimNum();
    for (size_t i = 0; i < size; i++) {
        if (shape->GetDim(i) == UNKNOWN_SHAPE_VALUE || shape->GetDim(i) == SHAPE_UNKNOWN_DIM_NUM) {
            return true;
        }
    }
    return false;
}

bool CheckUnknownShape(const gert::InferShapeContext *context) {
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
                                                        const GroupedMatmulCommonUtil &commonUtil) {
    if (xMDim_ < 0 || CheckUnknownShape(context)) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(CheckScenarioValidForShape(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckScenarioValidForShape failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeValid(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckShapeValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckFormatValid(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "CheckFormatValid failed."),
              return ge::GRAPH_FAILED);
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

ge::graphStatus GroupedMatmulWeightQuantChecker::InferOutShape(gert::InferShapeContext *context) const
{
    std::vector<int64_t> yDims = {xMDim_, weightNDim_};
    OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y, yDims) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Failed to update y shape."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckDtype(const gert::InferDataTypeContext *context) const
{
    auto xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto weightDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_WEIGHT, 0);
    // mandory param dtype check
    OP_CHECK_IF(X_TYPE_SUPPORT_SET.find(xDtype) == X_TYPE_SUPPORT_SET.end(),
              OP_LOGE(context->GetNodeName(), "Data type [%s] is not supported for x's 1st tensor.",
                        ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
              return ge::GRAPH_FAILED);

    OP_CHECK_IF(WEIGHT_TYPE_SUPPORT_SET.find(weightDtype) == WEIGHT_TYPE_SUPPORT_SET.end(),
              OP_LOGE(context->GetNodeName(), "Data type [%s] is not supported for weight.",
                        ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()),
              return ge::GRAPH_FAILED);

    auto antiquantScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    if (weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT) {
        OP_CHECK_IF(
            antiquantScaleDtype != ge::DT_FLOAT8_E8M0,
            OP_LOGE(context->GetNodeName(),
                      "Only support float8_e8m0 for antiquantScaleDataType when weight is fp4, but now it is [%s].",
                      ge::TypeUtils::DataTypeToAscendString(antiquantScaleDtype).GetString()),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(antiquantScaleDtype != xDtype,
                  OP_LOGE(context->GetNodeName(), "AntiquantScale datatype [%s] does not match xDtype [%s].",
                            ge::TypeUtils::DataTypeToAscendString(antiquantScaleDtype).GetString(),
                            ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
                  return ge::GRAPH_FAILED);
    }

    if (FP8_SUPPORT_SET.find(weightDtype) == FP8_SUPPORT_SET.end() || weightDtype == ge::DT_FLOAT4_E2M1 ||
        weightDtype == ge::DT_FLOAT) {
        auto antiquantOffsetDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_ANTIQUANT_OFFSET, 0);
        OP_CHECK_IF(antiquantOffsetDtype != xDtype,
                  OP_LOGE(context->GetNodeName(), "AntiquantOffset datatype [%s] does not match xDtype [%s].",
                            ge::TypeUtils::DataTypeToAscendString(antiquantOffsetDtype).GetString(),
                            ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
                  return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(BIAS_TYPE_SUPPORT_MAP.find(xDtype) == BIAS_TYPE_SUPPORT_MAP.end(),
              OP_LOGE(context->GetNodeName(), "Cannot find bias dtype match with xDtype [%s].",
                        ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
              return ge::GRAPH_FAILED);
    auto biasDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_BIAS, 0);
    OP_CHECK_IF(BIAS_TYPE_SUPPORT_MAP.at(xDtype).find(biasDtype) == BIAS_TYPE_SUPPORT_MAP.at(xDtype).end(),
              OP_LOGE(context->GetNodeName(), "Data type [%s] is not supported for bias, when xDtype is [%s].",
                        ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString(),
                        ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
              return ge::GRAPH_FAILED);

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

}  // namespace ops