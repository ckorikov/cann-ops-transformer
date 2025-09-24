/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_grouped_matmul_weight_quant_910_95_checker.h"

using namespace gmm;

namespace {
static const std::unordered_set<DataType> X_TYPE_SUPPORT_SET = {ge::DT_FLOAT16, ge::DT_BF16};
static const std::unordered_set<DataType> WEIGHT_TYPE_SUPPORT_SET = {
    ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8, ge::DT_FLOAT4_E2M1};
static const std::unordered_set<DataType> FP8_SUPPORT_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                                                                   ge::DT_HIFLOAT8};
} // namespace

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckTensorListDtype(const aclTensorList *tensorList,
                                                                            const DataType &xDtype,
                                                                            const DataType &weightDtype) const {
    if (tensorList != nullptr) {
        for (size_t i = 0; i < tensorList->Size(); i++) {
            const aclTensor *tensor = (*tensorList)[i];
            OP_CHECK_NULL(tensor, continue);
            if (weightDtype != ge::DT_FLOAT4_E2M1) {
                OP_CHECK_DTYPE_NOT_MATCH(tensor, xDtype, return ACLNN_ERR_PARAM_INVALID);
            } else {
                OP_CHECK_DTYPE_NOT_MATCH(tensor, ge::DT_FLOAT8_E8M0, return ACLNN_ERR_PARAM_INVALID);
            }
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckTensorListShape(const aclTensorList *tensorList,
                                                                            const std::string &tensorType) const
{
    // Check bias, antiquant scale, antiquant offset length, tensor's dims and shape under single weight condition
    if (tensorList == nullptr) {
        return ACLNN_SUCCESS;
    }

    uint64_t tensorSize = tensorList->Size();
    uint64_t weightGroupedSize = gmmParams_.weight->Size();
    // Check tensorList length matches weight.
    CHECK_COND(tensorSize == weightGroupedSize, ACLNN_ERR_PARAM_INVALID,
               "%s size[%lu] must be "
               "equal with weight size[%lu].",
               tensorType.c_str(), tensorSize, weightGroupedSize);

    auto tensor0Shape = (*tensorList)[0]->GetViewShape();
    auto w0Shape = (*gmmParams_.weight)[0]->GetViewShape();

    size_t tensorDimNum = tensor0Shape.GetDimNum();
    // Check tensor dimensions must be 2
    DataType tensorDtype = (*tensorList)[0]->GetDataType();
    if (tensorDtype != ge::DT_FLOAT8_E8M0) {
        // 2含义:仅支持antiquantscale和antiquantoffset的维度为2
        CHECK_COND(tensorDimNum == 2, ACLNN_ERR_PARAM_INVALID, "%s Dim must be 2, but now is [%zu].",
                   tensorType.c_str(), tensorDimNum);
    } else {
        // 当前仅伪量化的float8_e8m0类型的antiquantscale走到此分支，仅支持antiquantsacle维度为3
        CHECK_COND(tensorDimNum == 3, ACLNN_ERR_PARAM_INVALID,
                   "%s Dim must be 3 when the dtype is fp8_e8m0, but now is [%zu].", tensorType.c_str(), tensorDimNum);
    }

    // Check the first dimension, batch size must match the group size.
    uint64_t groupNum = w0Shape.GetDim(0);
    uint64_t batchSize = tensor0Shape.GetDim(0);
    CHECK_COND(batchSize == groupNum, ACLNN_ERR_PARAM_INVALID,
               "%s batch size[%lu] should be euqal "
               "with groupList length[%lu].",
               tensorType.c_str(), batchSize, groupNum);

    // Check tensor’s Ndim must match weight’s Ndim.
    uint64_t weightNDimIdx = w0Shape.GetDimNum() - 1;
    int64_t weightNDimValue = w0Shape.GetDim(weightNDimIdx);
    int64_t tensorNDimValue = tensor0Shape.GetDim(tensorDimNum - 1);
    CHECK_COND(tensorNDimValue == weightNDimValue, ACLNN_ERR_PARAM_INVALID,
               "NDim[%ld] of %s should be equal with NDim[%ld] of weight.", tensorNDimValue, tensorType.c_str(),
               weightNDimValue);

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckGmmQuantParamsEmpty() const
{
    CHECK_RET(gmmParams_.scaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmmParams_.offsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmmParams_.perTokenScaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckWeightFormatAndShape(const DataType &weightDtype) const
{
    // check single weight format and shape
    if (weightDtype != ge::DT_FLOAT4_E2M1) {
        if (WEIGHT_TYPE_SUPPORT_SET.find(weightDtype) != WEIGHT_TYPE_SUPPORT_SET.end()) {
            CHECK_COND(!op::IsPrivateFormat((*gmmParams_.weight)[0]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                       "The format of weight is invalid. It should only be ND when weight dtype is %s.",
                       op::ToString((*gmmParams_.weight)[0]->GetStorageFormat()).GetString());
        }
    } else {
        CHECK_COND(op::IsPrivateFormat((*gmmParams_.weight)[0]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                   "The format of weight is invalid. It should only be NZ when weight dtype is %s.",
                   op::ToString((*gmmParams_.weight)[0]->GetStorageFormat()).GetString());
    }

    size_t weightDimNum = (*gmmParams_.weight)[0]->GetViewShape().GetDimNum();
    CHECK_COND(weightDimNum == SPLIT_M_SINGLE_WEIGHT_DIM, ACLNN_ERR_PARAM_INVALID,
               "The weight dim num should be [%lu] in this case, but now is [%lu].", SPLIT_M_SINGLE_WEIGHT_DIM,
               weightDimNum);

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckTransposeStatus(const DataType &weightDtype) const {
    CHECK_COND(!gmmParams_.transposeX, ACLNN_ERR_PARAM_INVALID, "In weight quant case, x must not be transposed.");

    if (weightDtype != ge::DT_FLOAT4_E2M1) {
        if (WEIGHT_TYPE_SUPPORT_SET.find(weightDtype) != WEIGHT_TYPE_SUPPORT_SET.end()) {
            CHECK_COND(gmmParams_.transposeWeight, ACLNN_ERR_PARAM_INVALID,
                       "In weight quant case, when weight dtype is int8/fp8/hifloat8, weight must be transposed.");
        }
    } else {
        if (WEIGHT_TYPE_SUPPORT_SET.find(weightDtype) != WEIGHT_TYPE_SUPPORT_SET.end()) {
            CHECK_COND(!gmmParams_.transposeWeight, ACLNN_ERR_PARAM_INVALID,
                       "In weight quant case, when weight dtype is fp4, weight must be not transposed.");
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckNKValue() const
{
    auto w0Shape = (*gmmParams_.weight)[0]->GetViewShape();
    auto weightNIdx = w0Shape.GetDimNum() - 1;
    auto weightKIdx = w0Shape.GetDimNum() - 2;
    auto weightNDim = w0Shape.GetDim(weightNIdx);
    auto weightKDim = w0Shape.GetDim(weightKIdx);
    DataType weightDtype = (*gmmParams_.weight)[0]->GetDataType();

    CHECK_COND(weightNDim > 0 && weightNDim <= N_K_MAX_VALUE_WEIGHT_QUANT, ACLNN_ERR_PARAM_INVALID,
               "The n dim value should be positive and not larger than [%ld], but the actual value is [%ld].",
               N_K_MAX_VALUE_WEIGHT_QUANT, weightNDim);
    CHECK_COND(weightKDim > 0 && weightKDim <= N_K_MAX_VALUE_WEIGHT_QUANT, ACLNN_ERR_PARAM_INVALID,
               "The k dim value should be positive and not larger than [%ld], but the actual value is [%ld].",
               N_K_MAX_VALUE_WEIGHT_QUANT, weightKDim);
    if (weightDtype != ge::DT_FLOAT4_E2M1) {
        CHECK_COND((weightNDim % N_K_ALIGN_VALUE_WEIGHT_QUANT == 0) && (weightKDim % N_K_ALIGN_VALUE_WEIGHT_QUANT == 0),
                   ACLNN_ERR_PARAM_INVALID,
                   "The value of dim n, k should be an integer multiple of [%ld], but actual n is [%ld], k is [%ld].",
                   N_K_ALIGN_VALUE_WEIGHT_QUANT, weightNDim, weightKDim);
    } else {
        CHECK_COND((weightNDim % N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0) &&
                       (weightKDim % N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0),
                   ACLNN_ERR_PARAM_INVALID,
                   "The value of dim n, k should be an integer multiple of [%ld], but actual n is [%ld], k is [%ld].",
                   N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT, weightNDim, weightKDim);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckBiasDtype(const DataType &xDtype) const
{
    if (gmmParams_.biasOptional != nullptr) {
        DataType biasDtype = (*gmmParams_.biasOptional)[0]->GetDataType();
        if (xDtype == DataType::DT_BF16) {
            CHECK_COND(
                biasDtype == DataType::DT_BF16 || biasDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                "When xDtype is bfloat16, the bias dtype should be bfloat16 or float32, but the actual dtype is [%s].",
                op::ToString(biasDtype).GetString());
        } else if (xDtype == DataType::DT_FLOAT16) {
            CHECK_COND(biasDtype == DataType::DT_FLOAT16, ACLNN_ERR_PARAM_INVALID,
                       "When xDtype is float16, the bias dtype should be float16, but the actual dtype is [%s].",
                       op::ToString(biasDtype).GetString());
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckAntiQuantDtype(const DataType &xDtype,
                                                                           const DataType &weightDtype) const {
    CHECK_COND(CheckTensorListDtype(gmmParams_.antiquantScaleOptional, xDtype, weightDtype) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "AntiquantScale dtype does not match with x dtype [%s].",
               op::ToString(xDtype).GetString());
    if (weightDtype != ge::DT_FLOAT4_E2M1 && FP8_SUPPORT_SET.find(weightDtype) == FP8_SUPPORT_SET.end()) {
        CHECK_COND(CheckTensorListDtype(gmmParams_.antiquantOffsetOptional, xDtype, weightDtype) == ACLNN_SUCCESS,
                   ACLNN_ERR_PARAM_INVALID, "AntiquantOffset dtype does not match with x dtype [%s].",
                   op::ToString(xDtype).GetString());
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckAntiQuantShape() const
{
    DataType weightDtype = (*gmmParams_.weight)[0]->GetDataType();
    // 伪量化fp8、fp4场景不支持带有offset
    CHECK_COND(
        !(gmmParams_.antiquantOffsetOptional != nullptr &&
          (FP8_SUPPORT_SET.find(weightDtype) != FP8_SUPPORT_SET.end() || weightDtype == ge::DT_FLOAT4_E2M1)),
        ACLNN_ERR_PARAM_INVALID,
        "In weight quant case, it is unsupported for antiquantOffsetOptional to be non-nullptr when weightDtype is "
        "fp8/fp4.");
    CHECK_COND(CheckTensorListShape(gmmParams_.antiquantScaleOptional, "antiquantScale") == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_INVALID, "Invalid antiquantScale");
    if (weightDtype != ge::DT_FLOAT4_E2M1 &&
        FP8_SUPPORT_SET.find(weightDtype) == FP8_SUPPORT_SET.end()) {
        CHECK_COND(CheckTensorListShape(gmmParams_.antiquantOffsetOptional, "antiquantOffset") == ACLNN_SUCCESS,
                   ACLNN_ERR_PARAM_INVALID, "Invalid antiquantOffset");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckUnsupportApi(const DataType &weightDtype) const {
    // 拦截非支持接口调用伪量化int8的情况
    CHECK_COND(!(gmmParams_.apiVersion == GMMApiVersion::V1 && weightDtype == ge::DT_INT8),
               ACLNN_ERR_PARAM_INVALID, "Only AclnnGroupedMatmul V2/V3/V4/V5 supported int8 for weightDtype.");
    // 拦截非支持接口调用伪量化fp8的情况
    CHECK_COND(!(gmmParams_.apiVersion != GMMApiVersion::V5 && gmmParams_.apiVersion != GMMApiVersion::V4 &&
                 FP8_SUPPORT_SET.find(weightDtype) != FP8_SUPPORT_SET.end()),
               ACLNN_ERR_PARAM_INVALID, "Only AclnnGroupedMatmulV4/V5 supported fp8 for weightDtype.");
    // 拦截非支持接口调用伪量化fp4的情况, 通路仅支持FlOAT4_E2M1
    CHECK_COND(!(gmmParams_.apiVersion != GMMApiVersion::WeightNz && weightDtype == ge::DT_FLOAT4_E2M1),
               ACLNN_ERR_PARAM_INVALID, "Only AclnnGroupedMatmulWeightNz supported fp4 for weightDtype.");
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckGroupSize() const {
    CHECK_COND((*gmmParams_.antiquantScaleOptional)[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "The first element of antiquantScaleOptional must not be nullptr");
    auto antiquantScaleShape = (*gmmParams_.antiquantScaleOptional)[0]->GetViewShape();
    CHECK_COND((*gmmParams_.weight)[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "The first element of weight must not be nullptr");
    auto weightShape = (*gmmParams_.weight)[0]->GetViewShape();
    auto antiquantScaleDimNum = antiquantScaleShape.GetDimNum();
    int64_t groupSize = 0;
    // 3含义，当前shape为(g,k/groupsize,n)或者(g,n,k/groupSize)
    if (antiquantScaleDimNum == 3) {
        // 2含义: (g,k,n)的k轴索引
        int64_t kSize = gmmParams_.transposeWeight ? weightShape.GetDim(weightShape.GetDimNum() - 1)
                                                   : weightShape.GetDim(weightShape.GetDimNum() - 2);
        // 2含义: (g,k/groupSize,n)的k轴索引
        int64_t groupNum = gmmParams_.transposeWeight ? antiquantScaleShape.GetDim(antiquantScaleDimNum - 1)
                                                      : antiquantScaleShape.GetDim(antiquantScaleDimNum - 2);
        CHECK_COND(groupNum > 0, ACLNN_ERR_PARAM_INVALID,
                   "GroupNum must be greater than 0, but the actual groupNum is [%ld].", groupNum);
        CHECK_COND(kSize % groupNum == 0, ACLNN_ERR_PARAM_INVALID,
                   "kSize must be a multiple of groupNum, but the actual kSize is [%ld], groupNum is [%ld].", kSize,
                   groupNum);
        groupSize = kSize / groupNum;
    }
    // 当前伪量化仅支持groupsize为0或为32的整数倍
    CHECK_COND(groupSize % 32 == 0, ACLNN_ERR_PARAM_INVALID,
               "GroupSize must be a multiple of 32, but the actual groupSize is [%ld].", groupSize);
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuant91095Checker::CheckGroupedMatmulWeightQuant91095()
{
    DataType xDtype = gmmParams_.xDtype;
    DataType weightDtype = (*gmmParams_.weight)[0]->GetDataType();
    CHECK_COND(CheckGroupSize() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckGroupSize failed");
    CHECK_COND(gmmParams_.groupType == SPLIT_M, ACLNN_ERR_PARAM_INVALID,
               "Weight quant cases only support groupType 0 (split M), but the actual groupType is [%ld].",
               gmmParams_.groupType);

    CHECK_COND(gmmParams_.antiquantScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "AntiquantScale must not be nullptr in antiquant, but now is nullptr.");

    CHECK_COND(gmmParams_.x->Size() == 1 && gmmParams_.weight->Size() == 1 && gmmParams_.y->Size() == 1,
               ACLNN_ERR_PARAM_INVALID,
               "In weight quant case, the size of x, weight and y should all be 1, but the actual sizes are [%zu], "
               "[%zu] and [%zu].",
               gmmParams_.x->Size(), gmmParams_.weight->Size(), gmmParams_.y->Size());

    CHECK_COND((X_TYPE_SUPPORT_SET.find(xDtype) != X_TYPE_SUPPORT_SET.end()) &&
                   (WEIGHT_TYPE_SUPPORT_SET.find(weightDtype) != WEIGHT_TYPE_SUPPORT_SET.end()),
               ACLNN_ERR_PARAM_INVALID, "Weight quant case with x dtype [%s] and weight dtype [%s] is not supported.",
               op::ToString(xDtype).GetString(), op::ToString(weightDtype).GetString());

    CHECK_COND(CheckUnsupportApi(weightDtype) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckUnsupportApi failed.");

    CHECK_COND(CheckGmmQuantParamsEmpty() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected antiquant, but quant inputs are not empty!");

    CHECK_RET(CheckWeightFormatAndShape(weightDtype) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTransposeStatus(weightDtype) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNKValue() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckBiasDtype(xDtype) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAntiQuantDtype(xDtype, weightDtype) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAntiQuantShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
