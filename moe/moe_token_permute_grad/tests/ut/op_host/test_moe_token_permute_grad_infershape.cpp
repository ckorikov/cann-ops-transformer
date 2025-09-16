/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "common/utils/ut_op_common.h"

class MoeTokenPermuteGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenPermuteGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenPermuteGrad Proto Test TearDown" << std::endl;
  }
};

TEST_F(MoeTokenPermuteGrad, infershape_bf16) {
  gert::StorageShape permuted_out_d_shape = {{49152, 5120}, {49152, 5120}};
  gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
  // output
  gert::StorageShape input_grad_shape = {{6144, 5120}, {6144, 5120}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("MoeTokenPermuteGrad")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&permuted_out_d_shape, &sorted_indices_shape})
                    .OutputShapes({&input_grad_shape})
                    .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"num_topk", ge::AnyValue::CreateFrom<int64_t>(8)},
                      {"padded_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteGrad")->infer_shape;
  ge::graphStatus ret = infer_shape_func(context);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expectedInputGradShape = {6144, 5120};
  auto InputGradShape = context->GetOutputShape(0);
  EXPECT_EQ(ops::ToVector(*InputGradShape), expectedInputGradShape);
}

TEST_F(MoeTokenPermuteGrad, infertype_bf16) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteGrad"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteGrad")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_BF16;
    ge::DataType input_indices_ref = ge::DT_INT32;
    ge::DataType output_ref = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 1)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_ref, &input_indices_ref})
                              .OutputDataTypes({&output_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_ref);
    EXPECT_EQ(context->GetInputDataType(1), input_indices_ref);
    EXPECT_EQ(context->GetOutputDataType(0), output_ref);
  }
}

TEST_F(MoeTokenPermuteGrad, infershape_fp16) {
  gert::StorageShape permuted_out_d_shape = {{49152, 5120}, {49152, 5120}};
  gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
  // output
  gert::StorageShape input_grad_shape = {{6144, 5120}, {6144, 5120}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("MoeTokenPermuteGrad")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&permuted_out_d_shape, &sorted_indices_shape})
                    .OutputShapes({&input_grad_shape})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"num_topk", ge::AnyValue::CreateFrom<int64_t>(8)},
                      {"padded_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteGrad")->infer_shape;
  ge::graphStatus ret = infer_shape_func(context);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expectedInputGradShape = {6144, 5120};
  auto InputGradShape = context->GetOutputShape(0);
  EXPECT_EQ(ops::ToVector(*InputGradShape), expectedInputGradShape);
}

TEST_F(MoeTokenPermuteGrad, infertype_fp16) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteGrad"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteGrad")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_FLOAT16;
    ge::DataType input_indices_ref = ge::DT_INT32;
    ge::DataType output_ref = ge::DT_FLOAT16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 1)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_ref, &input_indices_ref})
                              .OutputDataTypes({&output_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_ref);
    EXPECT_EQ(context->GetInputDataType(1), input_indices_ref);
    EXPECT_EQ(context->GetOutputDataType(0), output_ref);
  }
}

TEST_F(MoeTokenPermuteGrad, infershape_fp32) {
  gert::StorageShape permuted_out_d_shape = {{49152, 5120}, {49152, 5120}};
  gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
  // output
  gert::StorageShape input_grad_shape = {{6144, 5120}, {6144, 5120}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("MoeTokenPermuteGrad")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&permuted_out_d_shape, &sorted_indices_shape})
                    .OutputShapes({&input_grad_shape})
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"num_topk", ge::AnyValue::CreateFrom<int64_t>(8)},
                      {"padded_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteGrad")->infer_shape;
  ge::graphStatus ret = infer_shape_func(context);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expectedInputGradShape = {6144, 5120};
  auto InputGradShape = context->GetOutputShape(0);
  EXPECT_EQ(ops::ToVector(*InputGradShape), expectedInputGradShape);
}

TEST_F(MoeTokenPermuteGrad, infertype_fp32) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteGrad"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteGrad")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_FLOAT;
    ge::DataType input_indices_ref = ge::DT_INT32;
    ge::DataType output_ref = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 1)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_ref, &input_indices_ref})
                              .OutputDataTypes({&output_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_ref);
    EXPECT_EQ(context->GetInputDataType(1), input_indices_ref);
    EXPECT_EQ(context->GetOutputDataType(0), output_ref);
  }
}