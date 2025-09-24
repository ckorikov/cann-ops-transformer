/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "common/utils/ut_op_common.h"

class MoeTokenPermuteWithEpGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenPermuteWithEpGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenPermuteWithEpGrad Proto Test TearDown" << std::endl;
  }
};

TEST_F(MoeTokenPermuteWithEpGrad, infershape_bf16) {
  std::vector<int64_t> range({0, 49152});
  gert::StorageShape permuted_tokens_output_d_shape = {{range[1] - range[0], 5120}, {range[1] - range[0], 5120}};
  gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
  gert::StorageShape permuted_probs_output_d_shape = {{range[1] - range[0]}, {range[1] - range[0]}};
  // output
  gert::StorageShape input_tokens_grad_shape = {{6144, 5120}, {6144, 5120}};
  gert::StorageShape input_probs_grad_shape = {{6144, 8}, {6144, 8}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("MoeTokenPermuteWithEpGrad")
                    .NodeIoNum(3, 2)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&permuted_tokens_output_d_shape, &sorted_indices_shape, &permuted_probs_output_d_shape})
                    .OutputShapes({&input_tokens_grad_shape, &input_probs_grad_shape})
                    .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"num_topk", ge::AnyValue::CreateFrom<int64_t>(8)},
                      {"range", ge::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
                      {"padded_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteWithEpGrad")->infer_shape;
  ge::graphStatus ret = infer_shape_func(context);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expectedInputGradShape = {6144, 5120};
  auto InputGradShape = context->GetOutputShape(0);
  EXPECT_EQ(ops::ToVector(*InputGradShape), expectedInputGradShape);
}

TEST_F(MoeTokenPermuteWithEpGrad, infertype_bf16) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteWithEpGrad"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteWithEpGrad")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_tokens_ref = ge::DT_BF16;
    ge::DataType input_indices_ref = ge::DT_INT32;
    ge::DataType input_probs_ref = ge::DT_BF16;
    ge::DataType output_tokens_ref = ge::DT_BF16;
    ge::DataType output_probs_ref = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 2)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_tokens_ref, &input_indices_ref, &input_probs_ref})
                              .OutputDataTypes({&output_tokens_ref, &output_probs_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_tokens_ref);
    EXPECT_EQ(context->GetInputDataType(1), input_indices_ref);
    EXPECT_EQ(context->GetOutputDataType(0), output_tokens_ref);
  }
}