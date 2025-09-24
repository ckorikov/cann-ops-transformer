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
 * \file test_moe_init_routing_proto.cpp
 * \brief
 */
#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "op_proto_test_util.h" // NOLINT
#include "fusion_ops.h"         // NOLINT
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class MoeInitRouting : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRouting SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRouting TearDown" << std::endl;
    }
};

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_01)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {-1, -1};
    std::vector<int64_t> expanded_idx_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_02)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {-1, -1};
    std::vector<int64_t> expanded_idx_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_03)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, 2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {-1, 2};
    std::vector<int64_t> expanded_idx_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_04)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {-1, -1};
    std::vector<int64_t> expanded_idx_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_05)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {-1, 3};
    std::vector<int64_t> expanded_idx_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_06)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {-1, -1};
    std::vector<int64_t> expanded_idx_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_07)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {-1, -1};
    std::vector<int64_t> expanded_idx_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_08)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, 2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {3, 2};
    std::vector<int64_t> expanded_idx_shape = {6};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_09)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {3, -1};
    std::vector<int64_t> expanded_idx_shape = {6};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_10)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {-1, 3};
    std::vector<int64_t> expanded_idx_shape = {-1};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_11)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 4}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 4}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {8, 3};
    std::vector<int64_t> expanded_idx_shape = {8};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_12)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 4}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 4}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {4, 3};
    std::vector<int64_t> expanded_idx_shape = {8};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_13)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 4}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 4}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 3);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto expanded_x_desc = op.GetOutputDesc("expanded_x");
    auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
    auto expanded_expert_idx_desc = op.GetOutputDesc("expanded_expert_idx");
    std::vector<int64_t> expanded_x_shape = {8, 3};
    std::vector<int64_t> expanded_idx_shape = {8};
    EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
    EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
    EXPECT_EQ(expanded_expert_idx_desc.GetShape().GetDims(), expanded_idx_shape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_14)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc = create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 5);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_15)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 5);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_16)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 5);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_17)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 5);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_18)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 5);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_19)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 5);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_20)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", -1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_21)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_22)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_23)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({3, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_24)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-1, 2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-1, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_25)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 5);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_26)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 5);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_27)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 5}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 5);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_28)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({3, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({3, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 5);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_29)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", -1);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_30)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc = create_desc_shape_range({2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_31)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_32)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_33)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_34)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3, 4}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_35)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 3, 4}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_36)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc = create_desc_shape_range({-3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_37)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({-3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_38)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({-3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_39)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, -3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_40)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, -3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_41)
{
    ge::op::MoeInitRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto x_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto row_idx_tensor_desc =
        create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    auto expert_idx_tensor_desc =
        create_desc_shape_range({2, -3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", x_tensor_desc);
    op.UpdateInputDesc("row_idx", row_idx_tensor_desc);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
    op.SetAttr("active_num", 2);
    Runtime2TestParam param{{"active_num"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_datatype_01)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRouting"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRouting")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        ge::DataType rstd_ref = ge::DT_INT32;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &rstd_ref, &rstd_ref})
                                  .OutputDataTypes({&output_ref, &rstd_ref, &rstd_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), rstd_ref);
        EXPECT_EQ(context->GetOutputDataType(2), rstd_ref);
    }
}