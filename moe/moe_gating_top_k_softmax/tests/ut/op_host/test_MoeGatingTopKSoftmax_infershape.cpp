/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "common/utils/ut_op_common.h"
#include "fusion_ops.h"

class MoeGatingTopKSoftmax : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKSoftmax Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKSoftmax Proto Test TearDown" << std::endl;
    }
};

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_legal_input)
{
    ge::op::MoeGatingTopKSoftmax op;
    op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT16));
    op.SetAttr("k", 2);
    Runtime2TestParam param{{"k"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_indices_desc = op.GetOutputDesc(1);
    auto output_source_row_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_output_shape = {4, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_indices_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_source_row_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_legal_input_2)
{
    ge::op::MoeGatingTopKSoftmax op;
    op.UpdateInputDesc("x", create_desc({4, 4, 4}, ge::DT_FLOAT16));
    op.SetAttr("k", 2);
    Runtime2TestParam param{{"k"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_indices_desc = op.GetOutputDesc(1);
    auto output_source_row_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_output_shape = {4, 4, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_indices_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_source_row_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_input_1)
{
    ge::op::MoeGatingTopKSoftmax op;
    op.UpdateInputDesc("x", create_desc({4, 4, 4, 4}, ge::DT_FLOAT16));
    op.SetAttr("k", 2);
    Runtime2TestParam param{{"k"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_input_2)
{
    ge::op::MoeGatingTopKSoftmax op;
    op.UpdateInputDesc("x", create_desc({4}, ge::DT_FLOAT16));
    op.SetAttr("k", 2);
    Runtime2TestParam param{{"k"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_k_1)
{
    ge::op::MoeGatingTopKSoftmax op;
    op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT16));
    op.SetAttr("k", -2);
    Runtime2TestParam param{{"k"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_k_2)
{
    ge::op::MoeGatingTopKSoftmax op;
    op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT16));
    op.SetAttr("k", 0);
    Runtime2TestParam param{{"k"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_k_3)
{
    ge::op::MoeGatingTopKSoftmax op;
    op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT16));
    op.SetAttr("k", 1025);
    Runtime2TestParam param{{"k"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_illegal_k_4)
{
    ge::op::MoeGatingTopKSoftmax op;
    op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT16));
    op.SetAttr("k", 5);
    Runtime2TestParam param{{"k"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_legal_dynamic_shape_1)
{
    ge::op::MoeGatingTopKSoftmax op;
    op.UpdateInputDesc("x", create_desc({-1, -1}, ge::DT_FLOAT16));
    op.SetAttr("k", 2);
    Runtime2TestParam param{{"k"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_indices_desc = op.GetOutputDesc(1);
    auto output_source_row_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_output_shape = {-1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_indices_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_source_row_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_legal_dynamic_shape_2)
{
    ge::op::MoeGatingTopKSoftmax op;
    op.UpdateInputDesc("x", create_desc({-1, -1, -1}, ge::DT_FLOAT16));
    op.SetAttr("k", 2);
    Runtime2TestParam param{{"k"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_indices_desc = op.GetOutputDesc(1);
    auto output_source_row_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_output_shape = {-1, -1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_indices_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_source_row_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infershape_diff_test_legal_dynamic_shape_3)
{
    ge::op::MoeGatingTopKSoftmax op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT16));
    op.SetAttr("k", 2);
    Runtime2TestParam param{{"k"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_indices_desc = op.GetOutputDesc(1);
    auto output_source_row_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_output_shape = {-2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_indices_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_source_row_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infer_datatype_01)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeGatingTopKSoftmax"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeGatingTopKSoftmax")->infer_datatype;

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
                                  .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref})
                                  .OutputDataTypes({&output_ref, &indices_ref, &indices_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), indices_ref);
        EXPECT_EQ(context->GetOutputDataType(2), indices_ref);
    }
}

TEST_F(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax_infer_datatype_02)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeGatingTopKSoftmax"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeGatingTopKSoftmax")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_FLOAT16;
        ge::DataType indices_ref = ge::DT_INT32;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(1, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref})
                                  .OutputDataTypes({&output_ref, &indices_ref, &indices_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), indices_ref);
        EXPECT_EQ(context->GetOutputDataType(2), indices_ref);
    }
}