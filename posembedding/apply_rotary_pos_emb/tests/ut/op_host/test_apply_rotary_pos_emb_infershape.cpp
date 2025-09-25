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
#include "nn_other.h"
#include "op_proto_test_util.h"
#include "fusion_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class ApplyRotaryPosEmbTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyRotaryPosEmbTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyRotaryPosEmbTest TearDown" << std::endl;
    }
};

TEST_F(ApplyRotaryPosEmbTest, apply_rotary_pos_emb_infer_shape_fp16)
{
    ge::op::ApplyRotaryPosEmb op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}, {2, 100}, {2, 100}, {2, 100}};
    auto tensor_desc = create_desc_shape_range(
        {-1, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {24, 1, 11, 128}, ge::FORMAT_ND, shape_range);
    std::vector<std::pair<int64_t, int64_t>> shape_range1 = {{2, 100}, {2, 100}, {2, 100}, {2, 100}};
    auto tensor_desc1 = create_desc_shape_range(
        {-1, -1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {24, 1, 1, 128}, ge::FORMAT_ND, shape_range1);
    op.UpdateInputDesc("query", tensor_desc);
    op.UpdateInputDesc("key", tensor_desc);
    op.UpdateInputDesc("cos", tensor_desc1);
    op.UpdateInputDesc("sin", tensor_desc1);
    op.SetAttr("layout", 0);
    op.SetAttr("rotary_mode", "half");
    Runtime2TestParam param{{"layout", "rotary_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto queryOutShape = op.GetOutputDescByName("query");
    std::vector<int64_t> expectedQueryOutshape = {-1, -1, -1, -1};
    EXPECT_EQ(queryOutShape.GetShape().GetDims(), expectedQueryOutshape);
}

TEST_F(ApplyRotaryPosEmbTest, apply_rotary_pos_emb_infershape_bf16)
{
    ge::op::ApplyRotaryPosEmb op;
    op.UpdateInputDesc("query", create_desc({4096, 4, 4, 128}, ge::DT_BF16));
    op.UpdateInputDesc("key", create_desc({4096, 4, 4, 128}, ge::DT_BF16));
    op.UpdateInputDesc("cos", create_desc({4096, 1, 1, 128}, ge::DT_BF16));
    op.UpdateInputDesc("sin", create_desc({4096, 1, 1, 128}, ge::DT_BF16));
    op.SetAttr("layout", 1);
    op.SetAttr("rotary_mode", "quarter");
    Runtime2TestParam param{{"layout", "rotary_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto keyOutShape = op.GetOutputDescByName("key");
    std::vector<int64_t> expectedKeyOutshape = {4096, 4, 4, 128};
    EXPECT_EQ(keyOutShape.GetShape().GetDims(), expectedKeyOutshape);
}

TEST_F(ApplyRotaryPosEmbTest, apply_rotary_pos_emb_infershape_empty_tensor_1)
{
    ge::op::ApplyRotaryPosEmb op;
    op.UpdateInputDesc("query", create_desc({2, 0, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("key", create_desc({2, 5, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("cos", create_desc({1, 1, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("sin", create_desc({1, 1, 4096, 120}, ge::DT_FLOAT));
    op.SetAttr("layout", 2);
    op.SetAttr("rotary_mode", "interleave");
    Runtime2TestParam param{{"layout", "rotary_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto queryOutShape = op.GetOutputDescByName("query");
    std::vector<int64_t> expectedQueryOutshape = {2, 0, 4096, 120};
    EXPECT_EQ(queryOutShape.GetShape().GetDims(), expectedQueryOutshape);
}

TEST_F(ApplyRotaryPosEmbTest, apply_rotary_pos_emb__infershape_empty_tensor_2)
{
    ge::op::ApplyRotaryPosEmb op;
    op.UpdateInputDesc("query", create_desc({2, 5, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("key", create_desc({2, 5, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("cos", create_desc({1, 0, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("sin", create_desc({1, 0, 4096, 120}, ge::DT_FLOAT));
    op.SetAttr("layout", 2);
    op.SetAttr("rotary_mode", "interleave");
    Runtime2TestParam param{{"layout", "rotary_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTest, apply_rotary_pos_emb_infershape_case_1)
{
    ge::op::ApplyRotaryPosEmb op;
    op.UpdateInputDesc("query", create_desc({-1, -1, -1, -1}, ge::DT_FLOAT));
    op.UpdateInputDesc("key", create_desc({2, 5, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("cos", create_desc({1, 1, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("sin", create_desc({1, 1, 4096, 120}, ge::DT_FLOAT));
    op.SetAttr("layout", 1);
    op.SetAttr("rotary_mode", "quarter");
    Runtime2TestParam param{{"layout", "rotary_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto keyOutShape = op.GetOutputDescByName("query");
    std::vector<int64_t> expectedKeyOutshape = {-1, -1, 4096, 120};
    EXPECT_EQ(keyOutShape.GetShape().GetDims(), expectedKeyOutshape);
}

TEST_F(ApplyRotaryPosEmbTest, apply_rotary_pos_emb_infershape_case_2)
{
    ge::op::ApplyRotaryPosEmb op;
    op.UpdateInputDesc("query", create_desc({2, 5, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("key", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("cos", create_desc({1, 1, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("sin", create_desc({1, 1, 4096, 120}, ge::DT_FLOAT));
    op.SetAttr("layout", 1);
    op.SetAttr("rotary_mode", "quarter");
    Runtime2TestParam param{{"layout", "rotary_mode"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto keyOutShape = op.GetOutputDescByName("key");
    std::vector<int64_t> expectedKeyOutshape = {-2};
    EXPECT_EQ(keyOutShape.GetShape().GetDims(), expectedKeyOutshape);
}

TEST_F(ApplyRotaryPosEmbTest, ApplyRotaryPosEmb_infershape_dtype_fp16_case)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyRotaryPosEmb"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyRotaryPosEmb")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"layout", ge::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeAttrs({{"rotary_mode", ge::AnyValue::CreateFrom<std::string>("half")}})
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetInputDataType(3), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
    }
}

TEST_F(ApplyRotaryPosEmbTest, ApplyRotaryPosEmb_infershape_dtype_bf16_case)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyRotaryPosEmb"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyRotaryPosEmb")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"layout", ge::AnyValue::CreateFrom<int64_t>(1)}})
                                  .NodeAttrs({{"rotary_mode", ge::AnyValue::CreateFrom<std::string>("interleave")}})
                                  .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetInputDataType(3), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
    }
}

TEST_F(ApplyRotaryPosEmbTest, ApplyRotaryPosEmb_infershape_dtype_float_case)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyRotaryPosEmb"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyRotaryPosEmb")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"layout", ge::AnyValue::CreateFrom<int64_t>(2)}})
                                  .NodeAttrs({{"rotary_mode", ge::AnyValue::CreateFrom<std::string>("quartar")}})
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetInputDataType(3), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
    }
}
