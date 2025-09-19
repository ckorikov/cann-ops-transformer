/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_init_routing_v3_proto.cpp
 * \brief
 */
#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "op_proto_test_util.h" // NOLINT
#include "experiment_ops.h"     // NOLINT
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class MoeInitRoutingV3 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV3 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV3 TearDown" << std::endl;
    }
};

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_01)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("active_num", -1);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 0);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {-1, -1};
    std::vector<int64_t> expanded_row_idx_shape = {-1};
    std::vector<int64_t> expert_tokens_count_shape = {7};
    std::vector<int64_t> expanded_scale_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_02)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 0);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {-1, -1};
    std::vector<int64_t> expanded_row_idx_shape = {-1};
    std::vector<int64_t> expert_tokens_count_shape = {7};
    std::vector<int64_t> expanded_scale_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_03)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({3, 128}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({3, 8}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 0);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {24, 128};
    std::vector<int64_t> expanded_row_idx_shape = {24};
    std::vector<int64_t> expert_tokens_count_shape = {7};
    std::vector<int64_t> expanded_scale_shape = {24};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_04)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", 1);
    op.SetAttr("row_idx_type", 0);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {-1, -1};
    std::vector<int64_t> expanded_row_idx_shape = {-1};
    std::vector<int64_t> expert_tokens_count_shape = {7};
    std::vector<int64_t> expanded_scale_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_05)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{8 * 512, 1024}};
    auto x_tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 0);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {-1, -1};
    std::vector<int64_t> expanded_row_idx_shape = {-1};
    std::vector<int64_t> expert_tokens_count_shape = {7};
    std::vector<int64_t> expanded_scale_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

// Exceptional case: when scale != (b*s), it should return ge::GRAPH_FAILED and report ERROR
TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_06)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({8 * 512, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    std::vector<std::pair<int64_t, int64_t>> expert_shape_range = {{8 * 512, 512}};
    auto expert_idx_tensor_desc =
        create_desc_shape_range({8 * 512, 512}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({8}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 0);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// scale=(end-start, )
TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_07)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({8 * 512, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    std::vector<std::pair<int64_t, int64_t>> expert_shape_range = {{8 * 512, 512}};
    auto expert_idx_tensor_desc =
        create_desc_shape_range({8 * 512, 512}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc = create_desc_shape_range(
        {
            7,
        },
        ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", 1);
    op.SetAttr("row_idx_type", 0);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {8 * 512 * 512, 1024};
    std::vector<int64_t> expanded_row_idx_shape = {8 * 512 * 512};
    std::vector<int64_t> expert_tokens_count_shape = {7};
    std::vector<int64_t> expanded_scale_shape = {8 * 512 * 512};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

// scale=(end-start, 1)
TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_08)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({8 * 512, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    std::vector<std::pair<int64_t, int64_t>> expert_shape_range = {{8 * 512, 512}};
    auto expert_idx_tensor_desc =
        create_desc_shape_range({8 * 512, 512}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({7, 1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", 0);
    op.SetAttr("row_idx_type", 0);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {8 * 512 * 512, 1024};
    std::vector<int64_t> expanded_row_idx_shape = {8 * 512 * 512};
    std::vector<int64_t> expert_tokens_count_shape = {7};
    std::vector<int64_t> expanded_scale_shape = {};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_09)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({8, 1024}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({8, 512}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc = create_desc_shape_range(
        {
            -1,
        },
        ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 0);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {8 * 512, 1024};
    std::vector<int64_t> expanded_row_idx_shape = {8 * 512};
    std::vector<int64_t> expert_tokens_count_shape = {7};
    std::vector<int64_t> expanded_scale_shape = {8 * 512};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_10)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc = create_desc_shape_range(
        {
            -1,
        },
        ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {87, 222});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {15114054, 192};
    std::vector<int64_t> expanded_row_idx_shape = {15114054};
    std::vector<int64_t> expert_tokens_count_shape = {135};
    std::vector<int64_t> expanded_scale_shape = {15114054};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_11)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({2087}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {87, 222});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {15114054, 192};
    std::vector<int64_t> expanded_row_idx_shape = {15114054};
    std::vector<int64_t> expert_tokens_count_shape = {135};
    std::vector<int64_t> expanded_scale_shape = {15114054};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

// int_64:  [-9223372036854775808, 9223372036854775807]
// uint_64: [                   0, 18446744073709551615]
TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_12)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc = create_desc_shape_range({9223372036854775807, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {32},
                                                 ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, 1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc = create_desc_shape_range(
        {
            -1,
        },
        ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {87, 222});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {9223372036854775807, 1};
    std::vector<int64_t> expanded_row_idx_shape = {9223372036854775807};
    std::vector<int64_t> expert_tokens_count_shape = {135};
    std::vector<int64_t> expanded_scale_shape = {9223372036854775807};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_13)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc = create_desc_shape_range(
        {
            -1,
        },
        ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {87, 222});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {15114054, 192};
    std::vector<int64_t> expanded_row_idx_shape = {15114054};
    std::vector<int64_t> expert_tokens_count_shape = {135};
    std::vector<int64_t> expanded_scale_shape = {15114054};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_14)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc = create_desc_shape_range(
        {
            -1,
        },
        ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {87, 222});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {15114054, 192};
    std::vector<int64_t> expanded_row_idx_shape = {15114054};
    std::vector<int64_t> expert_tokens_count_shape = {135};
    std::vector<int64_t> expanded_scale_shape = {15114054};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_15)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {87, 222});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {15114054, 192};
    std::vector<int64_t> expanded_row_idx_shape = {15114054};
    std::vector<int64_t> expert_tokens_count_shape = {135};
    std::vector<int64_t> expanded_scale_shape = {15114054};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_16)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({2087}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto offset_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.UpdateInputDesc("offset", offset_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {87, 222});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {15114054, 192};
    std::vector<int64_t> expanded_row_idx_shape = {15114054};
    std::vector<int64_t> expert_tokens_count_shape = {135};
    std::vector<int64_t> expanded_scale_shape = {15114054};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_17)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({2087}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto offset_tensor_desc =
        create_desc_shape_range({4}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.UpdateInputDesc("offset", offset_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {87, 222});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {15114054, 192};
    std::vector<int64_t> expanded_row_idx_shape = {15114054};
    std::vector<int64_t> expert_tokens_count_shape = {135};
    std::vector<int64_t> expanded_scale_shape = {15114054};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_18)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({135, 1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {87, 222});
    op.SetAttr("quant_mode", 0);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {15114054, 192};
    std::vector<int64_t> expanded_row_idx_shape = {15114054};
    std::vector<int64_t> expert_tokens_count_shape = {135};
    std::vector<int64_t> expanded_scale_shape = {};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
    EXPECT_EQ(expanded_scale_desc.GetShape().GetDims(), expanded_scale_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_19)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({135, 1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {87, 222});
    op.SetAttr("quant_mode", 0);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_21)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({4, 14}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({4, 5}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {0, 8});
    op.SetAttr("expert_tokens_num_type", 2);
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 0);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDescByName("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDescByName("expanded_row_idx");
    auto expert_tokens_count_desc = op.GetOutputDescByName("expert_tokens_count_or_cumsum");
    auto expanded_scale_desc = op.GetOutputDescByName("expanded_scale");
    std::vector<int64_t> expanded_x_shape = {20, 14};
    std::vector<int64_t> expanded_row_idx_shape = {20};
    std::vector<int64_t> expert_tokens_count_shape = {256, 2};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
    EXPECT_EQ(expert_tokens_count_desc.GetShape().GetDims(), expert_tokens_count_shape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_01)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType fp16_ref = ge::DT_FLOAT16;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        vector<int64_t> active_expert_range{1, 8};
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", ge::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"active_expert_range",
                                               ge::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&fp16_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&fp16_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), fp16_ref);
        printf("expended_x: %d", context->GetInputDataType(0));
        EXPECT_EQ(context->GetOutputDataType(0), fp16_ref);
        EXPECT_EQ(context->GetOutputDataType(1), int32_ref);
        EXPECT_EQ(context->GetOutputDataType(2), int64_ref);
        EXPECT_EQ(context->GetOutputDataType(4), fp32_ref);
    }
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_02)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType int8_ref = ge::DT_INT8;
        ge::DataType fp16_ref = ge::DT_FLOAT16;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        vector<int64_t> active_expert_range{1, 8};
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", ge::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_expert_range",
                                               ge::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&fp16_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&int8_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), fp16_ref);
        printf("expended_x: %d", context->GetInputDataType(0));
        EXPECT_EQ(context->GetOutputDataType(0), int8_ref);
        EXPECT_EQ(context->GetOutputDataType(1), int32_ref);
        EXPECT_EQ(context->GetOutputDataType(2), int64_ref);
        EXPECT_EQ(context->GetOutputDataType(4), fp32_ref);
    }
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_03)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType int8_ref = ge::DT_INT8;
        ge::DataType fp16_ref = ge::DT_FLOAT16;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        vector<int64_t> active_expert_range{1, 8};
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", ge::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", ge::AnyValue::CreateFrom<int64_t>(1)},
                                              {"active_expert_range",
                                               ge::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&fp16_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&int8_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), fp16_ref);
        printf("expended_x: %d", context->GetInputDataType(0));
        EXPECT_EQ(context->GetOutputDataType(0), int8_ref);
        EXPECT_EQ(context->GetOutputDataType(1), int32_ref);
        EXPECT_EQ(context->GetOutputDataType(2), int64_ref);
        EXPECT_EQ(context->GetOutputDataType(4), fp32_ref);
    }
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_04)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType int8_ref = ge::DT_INT8;
        ge::DataType fp16_ref = ge::DT_FLOAT16;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        vector<int64_t> active_expert_range{1, 7};
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", ge::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", ge::AnyValue::CreateFrom<int64_t>(1)},
                                              {"active_expert_range",
                                               ge::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", ge::AnyValue::CreateFrom<int64_t>(1)}})
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&fp16_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&int8_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), fp16_ref);
        printf("expended_x: %d", context->GetInputDataType(0));
        EXPECT_EQ(context->GetOutputDataType(0), int8_ref);
        EXPECT_EQ(context->GetOutputDataType(1), int32_ref);
        EXPECT_EQ(context->GetOutputDataType(2), int64_ref);
        EXPECT_EQ(context->GetOutputDataType(4), fp32_ref);
    }
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_05)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType int8_ref = ge::DT_INT8;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        vector<int64_t> active_expert_range{1, 7};
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 4)
                                  .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", ge::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"active_expert_range",
                                               ge::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", ge::AnyValue::CreateFrom<int64_t>(1)}})
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&int8_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&int8_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_datatype_06)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType int8_ref = ge::DT_INT8;
        ge::DataType fp32_ref = ge::DT_FLOAT;
        ge::DataType int32_ref = ge::DT_INT32;
        ge::DataType int64_ref = ge::DT_INT64;
        vector<int64_t> active_expert_range{1, 7};
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 4)
                                  .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"active_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_capacity", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"expert_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
                                              {"drop_pad_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_type", ge::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_num_flag", ge::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", ge::AnyValue::CreateFrom<int64_t>(1)},
                                              {"active_expert_range",
                                               ge::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
                                              {"row_idx_type", ge::AnyValue::CreateFrom<int64_t>(1)}})
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&int8_ref, &int32_ref, &fp32_ref})
                                  .OutputDataTypes({&int8_ref, &int32_ref, &int64_ref, &fp32_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infershape_range_00)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3"), nullptr);
    auto infer_shape_range_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3")->infer_shape_range;

    gert::Shape input_x_range_min{2, 2, 3, 8};
    gert::Shape input_x_range_max{2, -1, 3, 8};
    gert::Shape input_expert_idx_shape_range_min{2};
    gert::Shape input_expert_idx_shape_range_max{3};
    gert::Shape input_scale_shape_range_min{2};
    gert::Shape input_scale_shape_range_max{3};
    gert::Shape input_offset_shape_range_min{2};
    gert::Shape input_offset_shape_range_max{3};
    gert::Shape null1{};
    gert::Shape null2{};
    gert::Shape null3{};
    gert::Shape null4{};
    gert::Shape null5{};
    gert::Shape null6{};
    gert::Shape null7{};
    gert::Shape null8{};

    gert::Range<gert::Shape> input_x_range(&input_x_range_min, &input_x_range_max);
    gert::Range<gert::Shape> input_expert_idx_shape_range(&input_expert_idx_shape_range_min,
                                                          &input_expert_idx_shape_range_max);
    gert::Range<gert::Shape> input_scale_shape_range(&input_scale_shape_range_min, &input_scale_shape_range_max);
    gert::Range<gert::Shape> input_offset_shape_range(&input_offset_shape_range_min, &input_offset_shape_range_max);
    gert::Range<gert::Shape> output_0_shape_range(&null1, &null2);
    gert::Range<gert::Shape> output_1_shape_range(&null3, &null4);
    gert::Range<gert::Shape> output_2_shape_range(&null5, &null6);
    gert::Range<gert::Shape> output_3_shape_range(&null7, &null8);

    gert::Shape output_0_range_min{0, 0};
    gert::Shape output_0_range_max{-1, -1};
    gert::Range<gert::Shape> expect_output_0_shape_range(&output_0_range_min, &output_0_range_max);

    gert::Shape output_1_range_min{
        0,
    };
    gert::Shape output_1_range_max{
        -1,
    };
    gert::Range<gert::Shape> expect_output_1_shape_range(&output_1_range_min, &output_1_range_max);

    auto context_holder = gert::InferShapeRangeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(4, 4)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputShapeRanges({&input_x_range, &input_expert_idx_shape_range,
                                                 &input_scale_shape_range, &input_offset_shape_range})
                              .OutputShapeRanges({&output_0_shape_range, &output_1_shape_range, &output_2_shape_range,
                                                  &output_3_shape_range})
                              .Build();

    auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_range_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ops::ToString(context->GetOutputShapeRange(0)->GetMin()), ops::ToString(output_0_range_min));
    EXPECT_EQ(ops::ToString(context->GetOutputShapeRange(0)->GetMax()), ops::ToString(output_0_range_max));

    EXPECT_EQ(ops::ToString(context->GetOutputShapeRange(1)->GetMin()), ops::ToString(output_1_range_min));
    EXPECT_EQ(ops::ToString(context->GetOutputShapeRange(1)->GetMax()), ops::ToString(output_1_range_max));

    EXPECT_EQ(ops::ToString(context->GetOutputShapeRange(2)->GetMin()), ops::ToString(output_1_range_min));
    EXPECT_EQ(ops::ToString(context->GetOutputShapeRange(2)->GetMax()), ops::ToString(output_1_range_max));

    EXPECT_EQ(ops::ToString(context->GetOutputShapeRange(3)->GetMin()), ops::ToString(output_1_range_min));
    EXPECT_EQ(ops::ToString(context->GetOutputShapeRange(3)->GetMax()), ops::ToString(output_1_range_max));
}


TEST_F(MoeInitRoutingV3, moe_init_routing_v3_argerror_01)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({2087}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto offset_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.UpdateInputDesc("offset", offset_tensor_desc);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("expert_num", nullptr);
    op.SetAttr("quant_mode", -1);
    op.SetAttr("expert_tokens_num_type", 0);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_argerror_02)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({2087}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto offset_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.UpdateInputDesc("offset", offset_tensor_desc);
    op.SetAttr("expert_num", -1);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("expert_tokens_num_type", 0);
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_argerror_03)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({2087}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto offset_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.UpdateInputDesc("offset", offset_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("expert_tokens_num_type", nullptr);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_flag",
                             "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_argerror_04)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({2087}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto offset_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.UpdateInputDesc("offset", offset_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("expert_tokens_num_type", -1);
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_argerror_05)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({2087}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto offset_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.UpdateInputDesc("offset", offset_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("expert_tokens_num_type", 3);
    op.SetAttr("quant_mode", -1);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_argerror_06)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({2, 192}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto offset_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.UpdateInputDesc("offset", offset_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", 1);
    op.SetAttr("expert_tokens_num_type", 0);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_argerror_07)
{
    ge::op::MoeInitRoutingV3 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2087, 192}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2087, 7242}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto scale_tensor_desc =
        create_desc_shape_range({7, 191}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto offset_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.UpdateInputDesc("scale", scale_tensor_desc);
    op.UpdateInputDesc("offset", offset_tensor_desc);
    op.SetAttr("expert_num", 256);
    op.SetAttr("active_expert_range", {1, 8});
    op.SetAttr("quant_mode", 1);
    op.SetAttr("expert_tokens_num_type", 0);
    op.SetAttr("row_idx_type", 1);
    Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_tokens_num_type",
                             "expert_tokens_num_flag", "quant_mode", "active_expert_range", "row_idx_type"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}