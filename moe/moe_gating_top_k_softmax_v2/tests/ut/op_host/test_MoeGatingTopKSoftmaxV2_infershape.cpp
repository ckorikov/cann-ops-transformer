/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "common/utils/ut_op_common.h"
#include "fusion_ops.h"

class MoeGatingTopKSoftmaxV2 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2 Proto Test TearDown" << std::endl;
    }
};

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input)
{
    ge::op::MoeGatingTopKSoftmaxV2 op;
    op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT16));
    op.SetAttr("k", 2);
    op.SetAttr("renorm", 0);
    op.SetAttr("output_softmax_result_flag", true);
    Runtime2TestParam param{{"k", "renorm", "output_softmax_result_flag"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_indices_desc = op.GetOutputDesc(1);
    auto output_softmax_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_output_shape = {4, 2};
    std::vector<int64_t> expected_softmax_shape = {4, 4};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_indices_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_softmax_desc.GetShape().GetDims(), expected_softmax_shape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input1)
{
    ge::op::MoeGatingTopKSoftmaxV2 op;
    op.UpdateInputDesc("x", create_desc({4, 4, 3, 3}, ge::DT_FLOAT16));
    op.SetAttr("k", 2);
    op.SetAttr("renorm", 0);
    op.SetAttr("output_softmax_result_flag", true);
    Runtime2TestParam param{{"k", "renorm", "output_softmax_result_flag"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input2)
{
    ge::op::MoeGatingTopKSoftmaxV2 op;
    op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT16));
    op.SetAttr("k", 5);
    op.SetAttr("renorm", 0);
    op.SetAttr("output_softmax_result_flag", true);
    Runtime2TestParam param{{"k", "renorm", "output_softmax_result_flag"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input3)
{
    ge::op::MoeGatingTopKSoftmaxV2 op;
    op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT16));
    op.SetAttr("k", 2);
    op.SetAttr("renorm", 2);
    op.SetAttr("output_softmax_result_flag", true);
    Runtime2TestParam param{{"k", "renorm", "output_softmax_result_flag"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input4)
{
    ge::op::MoeGatingTopKSoftmaxV2 op;
    op.UpdateInputDesc("x", create_desc({-1, -1}, ge::DT_FLOAT16));
    op.SetAttr("k", 2);
    op.SetAttr("renorm", 1);
    op.SetAttr("output_softmax_result_flag", true);
    Runtime2TestParam param{{"k", "renorm", "output_softmax_result_flag"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infer_datatype_01)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeGatingTopKSoftmaxV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeGatingTopKSoftmaxV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        ge::DataType indices_ref = ge::DT_INT32;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(1, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref})
                                  .OutputDataTypes({&output_ref, &indices_ref, &indices_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), indices_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_ref);
    }
}
