/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_GMM_DSQ_BASE_H
#define ACLNN_GMM_DSQ_BASE_H

#include "aclnn_grouped_matmul_swiglu_quant_utils.h"

namespace gmm_dsq_base {

using namespace gmm_dsq;

constexpr int64_t SPLIT = 2L;
constexpr int64_t K_LIMIT_A8W8 = 65536L;
constexpr int64_t K_LIMIT_A8W4 = 20000L;
constexpr int64_t N_LIMIT = 10240L;
constexpr int64_t NZ_DIM_4_INT8 = 32L;
constexpr int64_t NZ_DIM_4_INT4 = 64L;
constexpr int64_t NZ_DIM_3 = 16L;
constexpr int64_t OUTPUT_IDX_0 = 0L;
constexpr int64_t OUTPUT_IDX_1 = 1L;
constexpr int64_t DIM_IDX_0 = 0L;
constexpr int64_t DIM_IDX_1 = 1L;
constexpr int64_t DIM_IDX_2 = 2L;
constexpr int64_t DIM_IDX_3 = 4L;
constexpr size_t X_DIM_LIMIT = 2UL;
constexpr size_t WEIGHT_SCALE_DIM_LIMIT = 2UL;
constexpr size_t WEIGHT_SCALE_PERGROUP_DIM_LIMIT = 3UL;
constexpr size_t WEIGHT_SCALE_PERCHANNEL_DIM_LIMIT = 2UL;
constexpr size_t TOKEN_SCALE_DIM_LIMIT = 1UL;
constexpr size_t BIAS_DIM_LIMIT = 2UL;
constexpr size_t GROUP_LIST_DIM_LIMIT = 1UL;
constexpr size_t QUANTOUT_DIM_LIMIT = 2UL;
constexpr size_t QUANTSCALEOUT_DIM_LIMIT = 1UL;
constexpr size_t INT4_PER_INT32 = 8UL;

const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
const std::initializer_list<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
const std::initializer_list<DataType> WEIGHT_SCALE_A8W4_DTYPE_SUPPORT_LIST = {DataType::DT_UINT64};
const std::initializer_list<DataType> X_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
const std::initializer_list<DataType> GROUP_LIST_DTYPE_SUPPORT_LIST = {DataType::DT_INT64};
const std::initializer_list<DataType> QUANTOUT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
const std::initializer_list<DataType> QUANTSCALEOUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
const std::initializer_list<DataType> WEIGHT_ASSIST_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

class GroupedMatmulSwigluQuantBaseHandler : public GroupedMatmulSwigluQuantHandler {
protected:
    bool CheckInputOutDimsA8W8()
    {
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.x, X_DIM_LIMIT, return false);
        size_t wLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* w = (*gmmDsqParams_.weight)[i];
            const aclTensor* wScale = (*gmmDsqParams_.weightScale)[i];
            op::Format weightViewFormat = w->GetViewFormat();
            if (IsPrivateFormat(weightViewFormat)) {
                OP_CHECK_WRONG_DIMENSION(w, WEIGHT_NZ_DIM_LIMIT, return false);
            } else {
                OP_CHECK_WRONG_DIMENSION(w, WEIGHT_ND_DIM_LIMIT, return false);
            }
            OP_CHECK_WRONG_DIMENSION(wScale, WEIGHT_SCALE_DIM_LIMIT, return false);
        }

        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.xScale, TOKEN_SCALE_DIM_LIMIT, return false);
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.groupList, GROUP_LIST_DIM_LIMIT, return false);
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.output, QUANTOUT_DIM_LIMIT, return false);
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.outputScale, QUANTSCALEOUT_DIM_LIMIT, return false);
        return true;
    }

    bool CheckSingleTensorListType(int64_t e, int64_t k, int64_t n)
    {
        // weight的NDshape期望为[E, K, N]
        op::Shape weightNDExpectShape1 = {e, k, n};
        // 单tesnsor weight的NZshape期望为[E, N // 32, K // 16, 16, 32]
        op::Shape weightNZExpectShape1 = {e, static_cast<int64_t>(n / NZ_DIM_4_INT8), static_cast<int64_t>(k / NZ_DIM_3),
                                        NZ_DIM_3, NZ_DIM_4_INT8};
        // weight的NDshape期望为[K, N]
        op::Shape weightNDExpectShape2 = {k, n};
        // weight的NZshape期望为[N // 32, K // 16, 16, 32]
        op::Shape weightNZExpectShape2 = {static_cast<int64_t>(n / NZ_DIM_4_INT8), static_cast<int64_t>(k / NZ_DIM_3),
                                        NZ_DIM_3, NZ_DIM_4_INT8};

        // weightScale的shape期望为[E, N]
        op::Shape weightScaleExpectShape1 = {e, n};
        op::Shape weightScaleExpectShape2 = {n};

        const aclTensor* w = (*gmmDsqParams_.weight)[0]; 
        const aclTensor* wScale = (*gmmDsqParams_.weightScale)[0];

        op::Format weightViewFormat = w->GetViewFormat();
        if (IsPrivateFormat(weightViewFormat)) {
            if (!(w->GetViewShape() == weightNZExpectShape1 || w->GetViewShape() == weightNZExpectShape2)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor for weight to have same size as %s or %s, but got %s.",
                        op::ToString(weightNZExpectShape1).GetString(),
                        op::ToString(weightNZExpectShape2).GetString(),
                        op::ToString(w->GetViewShape()).GetString());
                return false;
            }
        } else {
            if (!(w->GetViewShape() == weightNDExpectShape1 || w->GetViewShape() == weightNDExpectShape2)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor for weight to have same size as %s or %s, but got %s.",
                        op::ToString(weightNDExpectShape1).GetString(),
                        op::ToString(weightNDExpectShape2).GetString(),
                        op::ToString(w->GetViewShape()).GetString());
                return false;
            }
        }

        if (!(wScale->GetViewShape() == weightScaleExpectShape1 || wScale->GetViewShape() == weightScaleExpectShape2)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor for weight_scale to have same size as %s or %s, but got %s.",
                    op::ToString(weightScaleExpectShape1).GetString(),
                    op::ToString(weightScaleExpectShape2).GetString(),
                    op::ToString(wScale->GetViewShape()).GetString());
            return false;
        }
        return true;
    }

    bool CheckMultiTensorType(int64_t k, int64_t n)
    {
        // weight的NDshape期望为[K, N]
        op::Shape weightNDExpectShape = {k, n};
        // weight的NZshape期望为[N // 32, K // 16, 16, 32]
        op::Shape weightNZExpectShape = {static_cast<int64_t>(n / NZ_DIM_4_INT8), static_cast<int64_t>(k / NZ_DIM_3),
                                        NZ_DIM_3, NZ_DIM_4_INT8};

        // weightScale的shape期望为[N]
        op::Shape weightScaleExpectShape = {n};
        size_t wLength = gmmDsqParams_.weight->Size();

        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* w = (*gmmDsqParams_.weight)[0]; 
            const aclTensor* wScale = (*gmmDsqParams_.weightScale)[0];
            op::Format weightViewFormat = w->GetViewFormat();
            if (IsPrivateFormat(weightViewFormat)) {
                OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(w, weightNZExpectShape, return false);
            } else {
                OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(w, weightNDExpectShape, return false);
            }
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(wScale, weightScaleExpectShape, return false);
        }

        return true;
    }

    bool CheckTensorListShapeW8A8(int64_t e,int64_t k, int64_t n)
    {
        size_t wLength = gmmDsqParams_.weight->Size();
        if (wLength == static_cast<size_t>(1)) {
            return CheckSingleTensorListType(e, k, n);
        }

        return CheckMultiTensorType(k, n);
    }

    bool CheckInputOutShapeA8W8()
    {
        int64_t m = gmmDsqParams_.x->GetViewShape().GetDim(0);
        int64_t k = gmmDsqParams_.x->GetViewShape().GetDim(1);
        int64_t n = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDim(1);
        int64_t e = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0);
        if (n % SPLIT != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s, N is %ld , not an even number.", interfaceName_.c_str(), n);
            return false;
        }
        int64_t nAfterHalve = static_cast<int64_t>(n / SPLIT);
        // x的shape期望为[M, K]
        op::Shape xExpectShape = {m, k};
        // xScale的shape期望为[E, N]
        op::Shape xScaleExpectShape = {m};
        // output的shape期望为[M, N / 2]
        op::Shape outputExpectShape = {m, nAfterHalve};
        // outputScale的shape期望为[M]
        op::Shape outputScaleExpectShape = {m};

        auto ret = CheckTensorListShapeW8A8(e, k, n);
        if (!ret) {
            return false;
        }

        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.x, xExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.xScale, xScaleExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.output, outputExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.outputScale, outputScaleExpectShape, return false);
        // groupList的长度应小于等于weight的专家数
        int64_t groupListLen = gmmDsqParams_.groupList->GetViewShape().GetDim(0);
        if (groupListLen > e) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s A8W8, Length of 'groupList' out of range (expected to be in range of [1, "
                    "%ld], but got %ld)", interfaceName_.c_str(),
                    e, groupListLen);
            return false;
        }
        if (n > N_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s A8W8: The current version does not support the scenario that "
                    "N(%ld) is greater than %ld.", interfaceName_.c_str(),
                    n, N_LIMIT);
            return false;
        }
        if (k >= K_LIMIT_A8W8) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s A8W8, The current version does not support the scenario."
                    "The tail axis dimension of input0(x) is %ld, which need lower than %ld.",
                    interfaceName_.c_str(), k, K_LIMIT_A8W8);
            return false;
        }
        return true;
    }

    bool CheckInputOutDims() override
    {
        if (gmmDsqParams_.x->GetDataType() == DataType::DT_INT8
                && ((*gmmDsqParams_.weight)[0])->GetDataType() == DataType::DT_INT8) {
            return CheckInputOutDimsA8W8();
        }

        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckInputOutDims Failed: Only support weight and x is DT_INT8.");
        return false;
    }

    bool CheckInputOutShape() override
    {
        if (gmmDsqParams_.x->GetDataType() == DataType::DT_INT8
                && ((*gmmDsqParams_.weight)[0])->GetDataType() == DataType::DT_INT8) {
            return CheckInputOutShapeA8W8();
        }

        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckInputOutShape Failed: Only support weight and x is DT_INT8.");
        return false;
    }

    bool CheckDtypeValid() override
    {
        size_t wLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* wScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor* w = (*gmmDsqParams_.weight)[i];
        
            OP_CHECK_DTYPE_NOT_SUPPORT(w, WEIGHT_DTYPE_SUPPORT_LIST, return false);

            if (w->GetDataType() == DataType::DT_INT4) {
                OP_CHECK_DTYPE_NOT_SUPPORT(wScale, WEIGHT_SCALE_A8W4_DTYPE_SUPPORT_LIST, return false);
            } else if (w->GetDataType() == DataType::DT_INT8) {
                OP_CHECK_DTYPE_NOT_SUPPORT(wScale, WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
            }
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
        const aclTensor* w = (*gmmDsqParams_.weight)[0];
        bool isNZ = w->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ;
        if ((gmmDsqParams_.x->GetDataType() == DataType::DT_INT8
            && w->GetDataType() == DataType::DT_INT8) && !isNZ) {
            // fp16 in fp32 out that is split k template, not precision-advanced now
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s, The current version does not support the scenario."
                    "weight Format expect is FRACTAL_NZ, but got [%s].", interfaceName_.c_str(),
                    op::ToString(w->GetStorageFormat()).GetString());
            return false;
        }
        if (IsPrivateFormat(gmmDsqParams_.x->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s, The current version does not support the scenario."
                    "x Format Not support Private Format.", interfaceName_.c_str());
            return false;
        }
        if (IsPrivateFormat(gmmDsqParams_.output->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s, The current version does not support the scenario."
                    "output Format Not support Private Format.", interfaceName_.c_str());
            return false;
        }
        return true;
    }
};
}
#endif