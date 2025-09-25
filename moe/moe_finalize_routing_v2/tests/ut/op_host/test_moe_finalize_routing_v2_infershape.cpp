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
#include "fusion_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class MoeFinalizeRoutingV2Proto : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2Proto TearDown" << std::endl;
    }
};

TEST_F(MoeFinalizeRoutingV2Proto, error_shape_drop_less)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5, 1}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 0);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeFinalizeRoutingV2Proto, normal_shape_drop_less)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 0);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto out_shape = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_output_shape = {3, 5};
    EXPECT_EQ(out_shape.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5, 1}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape1)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 5);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape2)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc = create_desc_shape_range(
        {6, 1, 6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape3)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc = create_desc_shape_range(
        {6, 1, 6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 0);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape4)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6, 1}, ge::DT_INT32, ge::FORMAT_ND, {6, 1}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 0);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape5)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5, 1}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 0);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape6)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5, 1}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 0);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape7)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2, 1}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 0);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape8)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2, 1}, ge::DT_INT32, ge::FORMAT_ND, {3, 2, 1}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 0);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape11)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape12)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape13)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape14)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape15)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape16)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape17)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape18)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape19)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape20)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape21)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape22)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape23)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, error_shape24)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref1, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_err1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref1, &input_ref1, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_err2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_err3)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref1, &input_ref1, &input_ref, &input_ref, &input_ref, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_err4)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref1, &input_ref, &input_ref, &input_ref, &input_ref1, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_err5)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref1, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, normal_shape)
{
    ge::op::MoeFinalizeRoutingV2 op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 1, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_for_source_row_tensor_desc);

    op.SetAttr("drop_pad_mode", 1);

    Runtime2TestParam param{{"drop_pad_mode"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto out_shape = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_output_shape = {3, 5};
    EXPECT_EQ(out_shape.GetShape().GetDims(), expected_output_shape);
}