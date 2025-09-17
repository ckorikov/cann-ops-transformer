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

class MoeTokenPermuteWithEp : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithEp SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithEp TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteWithEp, infer_shape_01)
{
    gert::StorageShape tokens_shape = {{2, 5}, {2, 5}};
    gert::StorageShape indices_shape = {{2, 3}, {2, 3}};
    gert::StorageShape probs_shape = {{2, 3}, {2, 3}};
    gert::StorageShape permuted_tokens_shape = {{4, 5}, {4, 5}};
    gert::StorageShape sorted_indices_shape = {{6}, {6}};
    gert::StorageShape permuted_probs_shape = {{4}, {4}};
    std::vector<int64_t> range({1, 5});

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MoeTokenPermuteWithEp")
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&tokens_shape, &indices_shape, &probs_shape})
                      .OutputShapes({&permuted_tokens_shape, &sorted_indices_shape, &permuted_probs_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"range", ge::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
                           {"num_out_tokens", ge::AnyValue::CreateFrom<int64_t>(4)},
                           {"padded_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteWithEp")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_out_shape_0 = {4, 5};
    std::vector<int64_t> expected_out_shape_1 = {6};
    auto out_shape_0 = context->GetOutputShape(0);
    auto out_shape_1 = context->GetOutputShape(1);

    EXPECT_EQ(ops::ToVector(*out_shape_0), expected_out_shape_0);
    EXPECT_EQ(ops::ToVector(*out_shape_1), expected_out_shape_1);
}

TEST_F(MoeTokenPermuteWithEp, infer_shape_02)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteWithEp"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteWithEp")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_tokens_ref = ge::DT_BF16;
        ge::DataType input_indices_ref = ge::DT_INT64;
        ge::DataType input_probs_ref = ge::DT_BF16;
        ge::DataType output_tokens_ref = ge::DT_BF16;
        ge::DataType output_indices_ref = ge::DT_INT32;
        ge::DataType output_probs_ref = ge::DT_BF16;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 3)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_tokens_ref, &input_indices_ref, &input_probs_ref})
                                  .OutputDataTypes({&output_tokens_ref, &output_indices_ref, &output_probs_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_tokens_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_indices_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_tokens_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_indices_ref);
    }
}
