/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../../op_api/aclnn_quant_reduce_scatter.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class test_aclnn_quant_reduce_scatter : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "test_aclnn_quant_reduce_scatter SetUp" << endl; }

    static void TearDownTestCase() { cout << "test_aclnn_quant_reduce_scatter TearDown" << endl; }
};

struct QuantReduceScatterAclnnTestParam {
    string case_name;
    vector<int64_t> x_shape; // x数据shape
    vector<int64_t> scales_shape; // scales数据shape
    vector<int64_t> output_shape; // output数据shape
    aclDataType x_dtype; // x数据dtype
    aclDataType scales_dtype; // scales数据dtype
    aclDataType output_dtype; // 输出数据dtype
    aclnnStatus aclnn_status;
};

static QuantReduceScatterAclnnTestParam cases_params[] = {
    // 正常用例
    {"test_aclnn_quant_reduce_scatter_mx_H5120_right_one", {1024, 5120}, {1024, 80, 2}, {512, 5120}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_mx_H5120_right_two", {1024, 5120}, {1024, 80, 2}, {512, 5120}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_BF16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_mx_H7168_right_one", {1024, 7168}, {1024, 112, 2}, {512, 7168}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_mx_H7168_right_two", {1024, 7168}, {1024, 112, 2}, {512, 7168}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_BF16, ACLNN_SUCCESS},

    {"test_aclnn_quant_reduce_scatter_tg_H5120_right_one", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_INT8, ACL_FLOAT, ACL_FLOAT16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_H5120_right_two", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_HIFLOAT8, ACL_FLOAT, ACL_BF16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_H5120_right_three", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_H5120_right_four", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACLNN_SUCCESS},

    {"test_aclnn_quant_reduce_scatter_tg_H7168_right_one", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_INT8, ACL_FLOAT, ACL_FLOAT16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_H7168_right_two", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_HIFLOAT8, ACL_FLOAT, ACL_BF16, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_H7168_right_three", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_H7168_right_four", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACLNN_SUCCESS},
    {"test_aclnn_quant_reduce_scatter_tg_BS2048_right_one", {2048, 5120}, {2048, 40}, {1024, 5120}, ACL_INT8, ACL_FLOAT, ACL_FLOAT16, ACLNN_SUCCESS},

    // x类型异常
    {"test_aclnn_quant_reduce_scatter_mx_H5120_x_dtype_error_one", {1024, 5120}, {1024, 80, 2}, {512, 5120}, ACL_INT8, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID}, // mx量化时，x不应该为INT8,HIFLOAT8
    {"test_aclnn_quant_reduce_scatter_mx_H7168_x_dtype_error_one", {1024, 7168}, {1024, 112, 2}, {512, 7168}, ACL_INT8, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_mx_H5120_x_dtype_error_two", {1024, 5120}, {1024, 80, 2}, {512, 5120}, ACL_HIFLOAT8, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_mx_H7168_x_dtype_error_two", {1024, 7168}, {1024, 112, 2}, {512, 7168}, ACL_HIFLOAT8, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_tg_H5120_right_x_dtype_error_one", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID}, // T-G量化时，x不应该为FLOAT16
    {"test_aclnn_quant_reduce_scatter_tg_H7168_right_x_dtype_error_one", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID},

    {"test_aclnn_quant_reduce_scatter_mx_H5120_x_dtype_error_three", {1024, 5120}, {1024, 80, 2}, {512, 5120}, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID}, // x不应该为FLOAT
    {"test_aclnn_quant_reduce_scatter_mx_H7168_x_dtype_error_three", {1024, 7168}, {1024, 112, 2}, {512, 7168}, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    // scales类型异常
    {"test_aclnn_quant_reduce_scatter_H5120_scales_dtype_error_one", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_INT8, ACL_FLOAT16, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID}, // scales不支持FLOAT16
    {"test_aclnn_quant_reduce_scatter_H5120_scales_dtype_error_two", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_HIFLOAT8, ACL_FLOAT16, ACL_BF16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_H5120_scales_dtype_error_three", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_FLOAT8_E4M3FN, ACL_FLOAT16, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_H5120_scales_dtype_error_four", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_FLOAT8_E5M2, ACL_FLOAT16, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_H5120_scales_dtype_error_four", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID}, // scales不支持FLOAT8_E4M3FN

    {"test_aclnn_quant_reduce_scatter_H7168_scales_dtype_error_one", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_INT8, ACL_FLOAT16, ACL_FLOAT16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_H7168_scales_dtype_error_two", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_HIFLOAT8, ACL_FLOAT16, ACL_BF16, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_H7168_scales_dtype_error_three", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_FLOAT8_E4M3FN, ACL_FLOAT16, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_H7168_scales_dtype_error_four", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_FLOAT8_E5M2, ACL_FLOAT16, ACL_FLOAT, ACLNN_ERR_PARAM_INVALID},
    // output类型异常
    {"test_aclnn_quant_reduce_scatter_tg_H5120_output_dtype_error_one", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_INT8, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID}, // output不支持ACL_FLOAT8_E5M2
    {"test_aclnn_quant_reduce_scatter_tg_H5120_output_dtype_error_two", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_tg_H5120_output_dtype_error_three", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_tg_H5120_output_dtype_error_four", {1024, 5120}, {1024, 40}, {512, 5120}, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID},

    {"test_aclnn_quant_reduce_scatter_tg_H7168_output_dtype_error_one", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_INT8, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID}, // output不支持ACL_FLOAT8_E5M2
    {"test_aclnn_quant_reduce_scatter_tg_H7168_output_dtype_error_two", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_tg_H7168_output_dtype_error_three", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_reduce_scatter_tg_H7168_output_dtype_error_four", {1024, 7168}, {1024, 56}, {512, 7168}, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E5M2, ACLNN_ERR_PARAM_INVALID}
};

static void TestOneParamCase(const QuantReduceScatterAclnnTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;
    vector<int64_t> xShape = param.x_shape;
    vector<int64_t> scalesShape = param.scales_shape;
    vector<int64_t> outputShape = param.output_shape;
    aclDataType xDtype = param.x_dtype;
    aclDataType scalesDtype = param.scales_dtype;
    aclDataType outputDtype = param.output_dtype;
    aclnnStatus retStatus = param.aclnn_status;
    TensorDesc x = TensorDesc(xShape, xDtype, ACL_FORMAT_ND);
    TensorDesc scales = TensorDesc(scalesShape, scalesDtype, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc(outputShape, outputDtype, ACL_FORMAT_ND);
    const char* reduceOp = "sum";
    auto ut = OP_API_UT(aclnnQuantReduceScatter,
                        INPUT(x, scales, "test_quant_reduce_scatter", reduceOp),
                        OUTPUT(output));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, retStatus);
}

TEST_F(test_aclnn_quant_reduce_scatter, cases_params)
{
    if (std::size(cases_params) != 0) {
        uint64_t numCases = sizeof(cases_params) / sizeof(cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(cases_params[idx]);
        }
    }
}