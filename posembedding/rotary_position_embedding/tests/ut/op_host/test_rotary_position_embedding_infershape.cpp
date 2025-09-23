/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "array_ops.h"
#include "experiment_ops.h"
#include "nn_other.h"
#include "common/utils/ut_op_common.h"

class RotaryPositionEmbedding : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RotaryPositionEmbedding Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RotaryPositionEmbedding Proto Test TearDown" << std::endl;
    }
};

TEST_F(RotaryPositionEmbedding, RotaryPositionEmbedding_infershape_case_0)
{
    ge::op::RotaryPositionEmbedding op;
    op.UpdateInputDesc("x", create_desc({1, 64, 2, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("cos", create_desc({1, 64, 1, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("sin", create_desc({1, 64, 1, 64}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

TEST_F(RotaryPositionEmbedding, RotaryPositionEmbedding_infershape_case_1)
{
    ge::op::RotaryPositionEmbedding op;
    op.UpdateInputDesc("x", create_desc({4096, 4, 4, 128}, ge::DT_BF16));
    op.UpdateInputDesc("cos", create_desc({4096, 1, 1, 128}, ge::DT_BF16));
    op.UpdateInputDesc("sin", create_desc({4096, 1, 1, 128}, ge::DT_BF16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

TEST_F(RotaryPositionEmbedding, RotaryPositionEmbedding_infershape_case_2)
{
    ge::op::RotaryPositionEmbedding op;
    op.UpdateInputDesc("x", create_desc({2, 5, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("cos", create_desc({1, 1, 4096, 120}, ge::DT_FLOAT));
    op.UpdateInputDesc("sin", create_desc({1, 1, 4096, 120}, ge::DT_FLOAT));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

TEST_F(RotaryPositionEmbedding, RotaryPositionEmbedding_infershape_mode_0_fp16_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"mode", ge::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(RotaryPositionEmbedding, RotaryPositionEmbedding_infershape_mode_0_bf16_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"mode", ge::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(RotaryPositionEmbedding, RotaryPositionEmbedding_infershape_mode_0_fp32_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"mode", ge::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(RotaryPositionEmbedding, RotaryPositionEmbedding_infershape_mode_1_fp16_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"mode", ge::AnyValue::CreateFrom<int64_t>(1)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(RotaryPositionEmbedding, RotaryPositionEmbedding_infershape_mode_1_bf16_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"mode", ge::AnyValue::CreateFrom<int64_t>(1)}})
                                  .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(RotaryPositionEmbedding, RotaryPositionEmbedding_infershape_mode_1_fp32_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotaryPositionEmbedding")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"mode", ge::AnyValue::CreateFrom<int64_t>(1)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}