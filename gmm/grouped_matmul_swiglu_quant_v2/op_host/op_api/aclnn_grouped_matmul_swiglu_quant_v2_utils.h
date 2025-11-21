/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_V2_UTILS_H
#define OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_V2_UTILS_H

#include "aclnn_grouped_matmul_swiglu_quant_utils.h"

namespace gmmSwigluQuantV2 {

using namespace gmm_dsq;

constexpr int64_t OUTPUT_IDX_0 = 0L;
constexpr int64_t OUTPUT_IDX_1 = 1L;
constexpr size_t MX_SPLIT_K_PER_TOKEN_SCALE_DIM = 3UL;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_THIRD_DIM_INDEX = 3;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2L;
constexpr size_t MX_SPLIT_M_SCALE_DIM = 4UL;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2L;
constexpr int64_t SWIGLU_SPLIT_SIZE = 64L;

const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN,
                                                              DataType::DT_FLOAT8_E5M2};
const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN,
                                                                   DataType::DT_FLOAT8_E5M2};
const std::initializer_list<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
const std::initializer_list<DataType> X_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
const std::initializer_list<DataType> GROUP_LIST_DTYPE_SUPPORT_LIST = {DataType::DT_INT64};
const std::initializer_list<DataType> QUANTOUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN,
                                                                     DataType::DT_FLOAT8_E5M2};
const std::initializer_list<DataType> QUANTSCALEOUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
class GroupedMatmulSwigluQuantBaseHandler : public GroupedMatmulSwigluQuantHandler {
protected:

    bool IsTransposeForMxShape(const aclTensor *tensor)
    {
        auto shape = tensor->GetViewShape();
        if (shape.GetDimNum() < MX_SPLIT_K_PER_TOKEN_SCALE_DIM) {
            return false;
        }
        int64_t firstLastDim = shape.GetDimNum() - 1;
        int64_t secondLastDim = shape.GetDimNum() - LAST_SECOND_DIM_INDEX;
        int64_t thirdLastDim = shape.GetDimNum() - LAST_THIRD_DIM_INDEX;
        auto strides = tensor->GetViewStrides();
        if (strides[firstLastDim] == 1 && strides[thirdLastDim] == MXFP_MULTI_BASE_SIZE &&
            strides[secondLastDim] == shape.GetDim(thirdLastDim) * MXFP_MULTI_BASE_SIZE) {
            return true;
        }
        return false;
    }

    bool IsTransposeLastTwoDims(const aclTensor *tensor)
    {
        auto shape = tensor->GetViewShape();
        int64_t dim1 = shape.GetDimNum() - 1;
        int64_t dim2 = shape.GetDimNum() - 2;
        auto strides = tensor->GetViewStrides();
        if (strides[dim2] == 1 && strides[dim1] == shape.GetDim(dim2)) {
            int64_t tmpNxD = shape.GetDim(dim1) * shape.GetDim(dim2);
            for (int64_t batchDim = shape.GetDimNum() - 3; batchDim >= 0; batchDim--) {
                if (strides[batchDim] != tmpNxD) {
                    return false;
                }
                tmpNxD *= shape.GetDim(batchDim);
            }
            return true;
        }
        return false;
    }

    void CreateContiguousTensorListForMXTypeMScale(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                                aclOpExecutor *executor)
    {
        op::Shape shape;
        for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
            const aclTensor *inputTensor = (*tensorList)[idx];
            op::Shape viewShape = inputTensor->GetViewShape();
            shape.SetScalar();
            if (viewShape.GetDimNum() < MX_SPLIT_M_SCALE_DIM) {
                continue;
            }
            shape.AppendDim(viewShape.GetDim(0));
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - LAST_SECOND_DIM_INDEX));
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - LAST_THIRD_DIM_INDEX));
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - 1));
            aclTensor *tensor =
                executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
            tensor->SetStorageFormat(inputTensor->GetStorageFormat());
            newTensorList.emplace_back(tensor);
        }
    }

    void CreateContiguousTensorList(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                    aclOpExecutor *executor)
    {
        op::Shape shape;
        for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
            const aclTensor *inputTensor = (*tensorList)[idx];
            op::Shape viewShape = inputTensor->GetViewShape();
            uint32_t viewShapeDimsNum = viewShape.GetDimNum();
            shape.SetScalar();
            // 2: the second last dimension; in for-loops, it indicates dimensions before the second last remain unchanged.
            for (uint32_t i = 0; i < viewShapeDimsNum - 2; ++i) {
                shape.AppendDim(viewShape.GetDim(i));
            }
            // viewShapeDimsNum - 1, the dim value of the last dim. viewShapeDimsNum - 2, the dim value of the second last
            // dim.
            shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 1));
            shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 2)); // 2:the second last dim.
            aclTensor *tensor =
                executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
            tensor->SetStorageFormat(inputTensor->GetStorageFormat());
            newTensorList.emplace_back(tensor);
        }
    }

    bool CheckInputOutDims() override
    {
        return true;
    }

    bool CheckInputOutShape() override
    {
        // 判断weight和weightScale是否转置，是则对两者进行转置动作
        bool transposeweightScale = IsTransposeForMxShape((*gmmDsqParams_.weightScale)[0]);
        bool transposeweight = IsTransposeLastTwoDims((*gmmDsqParams_.weight)[0]);
        if (transposeweightScale && transposeweight) {
            gmmDsqParams_.transposeWeight = true;
            auto uniqueExecutor = CREATE_EXECUTOR();
            CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
            aclOpExecutor *executorPtr = uniqueExecutor.get();
            CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
            std::vector<aclTensor *> scaleTensorList;
            std::vector<aclTensor *> weightTensorList;
            CreateContiguousTensorListForMXTypeMScale(gmmDsqParams_.weightScale, scaleTensorList, executorPtr);
            gmmDsqParams_.weightScale = executorPtr->AllocTensorList(scaleTensorList.data(), scaleTensorList.size());
            CreateContiguousTensorList(gmmDsqParams_.weight, weightTensorList, executorPtr);
            gmmDsqParams_.weight = executorPtr->AllocTensorList(weightTensorList.data(), weightTensorList.size());
            uniqueExecutor.ReleaseTo(executor_);
        }
        return true;
    }

    bool CheckDtypeValid() override
    {
        size_t wLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* wScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor* w = (*gmmDsqParams_.weight)[i];      
            OP_CHECK_DTYPE_NOT_SUPPORT(w, WEIGHT_DTYPE_SUPPORT_LIST, return false);
            OP_CHECK_DTYPE_NOT_SUPPORT(wScale, WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
        }
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.x, X_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.xScale, X_SCALE_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.groupList, GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.output, QUANTOUT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.outputScale, QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);

        return true;
    }

    bool CheckFormat() override
    {
        size_t wLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* wScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor* w = (*gmmDsqParams_.weight)[i];
            CHECK_COND(w->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
                "Format of weight should be ND, current format is %s",
               op::ToString(w->GetStorageFormat()).GetString());
            CHECK_COND(wScale->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
                "Format of weightScale should be ND, current format is %s",
               op::ToString(wScale->GetStorageFormat()).GetString());
        }
        CHECK_COND(gmmDsqParams_.x->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "Format of x should be ND, current format is format is %s",
               op::ToString(gmmDsqParams_.x->GetStorageFormat()).GetString());
        CHECK_COND(gmmDsqParams_.xScale->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
                "Format of xScale should be ND, current format is %s",
               op::ToString(gmmDsqParams_.xScale->GetStorageFormat()).GetString());
        CHECK_COND(gmmDsqParams_.groupList->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
                "Format of groupList should be ND, current format is  %s",
               op::ToString(gmmDsqParams_.groupList->GetStorageFormat()).GetString());
        CHECK_COND(gmmDsqParams_.output->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
                "Format of output should be ND, current format is %s",
               op::ToString(gmmDsqParams_.output->GetStorageFormat()).GetString());
        CHECK_COND(gmmDsqParams_.outputScale->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
                "Format of outputScale should be ND, current format is %s",
               op::ToString(gmmDsqParams_.outputScale->GetStorageFormat()).GetString());
        
        return true;
    }
};
}
#endif