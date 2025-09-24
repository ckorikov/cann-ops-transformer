/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_gating_top_k_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "op_proto_test_util.h" // NOLINT
#include "experiment_ops.h"     // NOLINT
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class MoeGatingTopK : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopK SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopK TearDown" << std::endl;
    }
};

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_01)
{
    ge::op::MoeGatingTopK op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({16, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto bias_tensor_desc = create_desc_shape_range(
        {
            256,
        },
        ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("bias", bias_tensor_desc);
    op.SetAttr("k", 8);
    op.SetAttr("k_group", 4);
    op.SetAttr("group_count", 8);
    Runtime2TestParam param{{"k", "k_group", "group_count"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto y_desc = op.GetOutputDesc("y");
    auto expert_idx_desc = op.GetOutputDesc("expert_idx");
    auto out_desc = op.GetOutputDesc("out");
    std::vector<int64_t> y_shape = {16, 8};
    std::vector<int64_t> expert_idx_shape = {16, 8};
    std::vector<int64_t> out_shape = {16, 256};
    EXPECT_EQ(y_desc.GetShape().GetDims(), y_shape);
    EXPECT_EQ(expert_idx_desc.GetShape().GetDims(), expert_idx_shape);
    EXPECT_EQ(out_desc.GetShape().GetDims(), out_shape);
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_02)
{
    ge::op::MoeGatingTopK op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto bias_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("bias", bias_tensor_desc);
    op.SetAttr("k", 8);
    op.SetAttr("k_group", 4);
    op.SetAttr("group_count", 8);
    Runtime2TestParam param{{"k", "k_group", "group_count"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto y_desc = op.GetOutputDesc("y");
    auto expert_idx_desc = op.GetOutputDesc("expert_idx");
    auto out_desc = op.GetOutputDesc("out");
    std::vector<int64_t> y_shape = {-1, 8};
    std::vector<int64_t> expert_idx_shape = {-1, 8};
    std::vector<int64_t> out_shape = {-1, -1};
    EXPECT_EQ(y_desc.GetShape().GetDims(), y_shape);
    EXPECT_EQ(expert_idx_desc.GetShape().GetDims(), expert_idx_shape);
    EXPECT_EQ(out_desc.GetShape().GetDims(), out_shape);
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_03)
{
    ge::op::MoeGatingTopK op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto bias_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("bias", bias_tensor_desc);
    op.SetAttr("k", 8);
    op.SetAttr("k_group", 4);
    op.SetAttr("group_count", 8);
    Runtime2TestParam param{{"k", "k_group", "group_count"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto y_desc = op.GetOutputDesc("y");
    auto expert_idx_desc = op.GetOutputDesc("expert_idx");
    auto out_desc = op.GetOutputDesc("out");
    std::vector<int64_t> y_shape = {-1, 8};
    std::vector<int64_t> expert_idx_shape = {-1, 8};
    std::vector<int64_t> out_shape = {-1, -1};
    EXPECT_EQ(y_desc.GetShape().GetDims(), y_shape);
    EXPECT_EQ(expert_idx_desc.GetShape().GetDims(), expert_idx_shape);
    EXPECT_EQ(out_desc.GetShape().GetDims(), out_shape);
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_shape_04)
{
    ge::op::MoeGatingTopK op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto bias_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("bias", bias_tensor_desc);
    op.SetAttr("k", 8);
    op.SetAttr("k_group", 4);
    op.SetAttr("group_count", 8);
    Runtime2TestParam param{{"k", "k_group", "group_count"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto y_desc = op.GetOutputDesc("y");
    auto expert_idx_desc = op.GetOutputDesc("expert_idx");
    auto out_desc = op.GetOutputDesc("out");
    std::vector<int64_t> y_shape = {-1, 8};
    std::vector<int64_t> expert_idx_shape = {-1, 8};
    std::vector<int64_t> out_shape = {-1, -1};
    EXPECT_EQ(y_desc.GetShape().GetDims(), y_shape);
    EXPECT_EQ(expert_idx_desc.GetShape().GetDims(), expert_idx_shape);
    EXPECT_EQ(out_desc.GetShape().GetDims(), out_shape);
}

TEST_F(MoeGatingTopK, moe_gating_top_k_infer_datatype_01)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeGatingTopK"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeGatingTopK")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        ge::DataType rstd_ref = ge::DT_INT32;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &rstd_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), rstd_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_ref);
    }
}
