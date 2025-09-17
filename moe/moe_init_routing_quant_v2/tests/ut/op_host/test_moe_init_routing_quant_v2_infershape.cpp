/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>  // NOLINT
#include <iostream>
#include "op_proto_test_util.h"  // NOLINT
#include "fusion_ops.h"          // NOLINT
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class MoeInitRoutingQuantV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeInitRoutingQuantV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeInitRoutingQuantV2 TearDown" << std::endl;
  }
};

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_01) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto idx_tensor_desc =
      create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 10);
  op.SetAttr("expert_capacity", 6);
  op.SetAttr("expert_num", 4);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 0);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 0);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto expanded_x_desc = op.GetOutputDesc("expanded_x");
  auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
  auto expert_tokens_count_or_cumsum_desc = op.GetOutputDesc("expert_tokens_count_or_cumsum");
  auto expert_tokens_before_capacity_desc = op.GetOutputDesc("expert_tokens_before_capacity");
  std::vector<int64_t> expanded_x_shape = {-1, -1};
  std::vector<int64_t> expanded_row_idx_shape = {-1};
  std::vector<int64_t> expert_tokens_count_or_cumsum_shape = {};
  std::vector<int64_t> expert_tokens_before_capacity_shape = {};
  EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
  EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
  EXPECT_EQ(expert_tokens_count_or_cumsum_desc.GetShape().GetDims(), expert_tokens_count_or_cumsum_shape);
  EXPECT_EQ(expert_tokens_before_capacity_desc.GetShape().GetDims(), expert_tokens_before_capacity_shape);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_02) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc =
      create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto idx_tensor_desc =
      create_desc_shape_range({-1, -1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({-1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 1);
  op.SetAttr("expert_capacity", 6);
  op.SetAttr("expert_num", 4);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 0);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 1);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto expanded_x_desc = op.GetOutputDesc("expanded_x");
  auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
  auto expert_tokens_count_or_cumsum_desc = op.GetOutputDesc("expert_tokens_count_or_cumsum");
  auto expert_tokens_before_capacity_desc = op.GetOutputDesc("expert_tokens_before_capacity");
  auto dynamic_quant_scale_desc = op.GetOutputDesc("dynamic_quant_scale");
  std::vector<int64_t> expanded_x_shape = {-1, -1};
  std::vector<int64_t> expanded_row_idx_shape = {-1};
  std::vector<int64_t> expert_tokens_count_or_cumsum_shape = {};
  std::vector<int64_t> dynamic_quant_scale_shape = {-1};
  EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
  EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_row_idx_shape);
  EXPECT_EQ(expert_tokens_count_or_cumsum_desc.GetShape().GetDims(), expert_tokens_count_or_cumsum_shape);
  EXPECT_EQ(expert_tokens_before_capacity_desc.GetShape().GetDims(), expert_tokens_count_or_cumsum_shape);
  EXPECT_EQ(dynamic_quant_scale_desc.GetShape().GetDims(), dynamic_quant_scale_shape);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_03) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto idx_tensor_desc = create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 20);
  op.SetAttr("expert_capacity", 2);
  op.SetAttr("expert_num", 4);
  op.SetAttr("drop_pad_mode", 1);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 0);
  op.SetAttr("expert_tokens_before_capacity_flag", true);
  op.SetAttr("quant_mode", 0);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto expanded_x_desc = op.GetOutputDesc("expanded_x");
  auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
  auto expert_tokens_before_capacity_desc = op.GetOutputDesc("expert_tokens_before_capacity");
  std::vector<int64_t> expanded_x_shape = {4, 2, 5};
  std::vector<int64_t> expanded_idx_shape = {6};
  std::vector<int64_t> expert_tokens_before_capacity_shape = {4};
  EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
  EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
  EXPECT_EQ(expert_tokens_before_capacity_desc.GetShape().GetDims(), expert_tokens_before_capacity_shape);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_04) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto idx_tensor_desc = create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 3);
  op.SetAttr("expert_capacity", 1);
  op.SetAttr("expert_num", 1);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 1);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 1);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto expanded_x_desc = op.GetOutputDesc("expanded_x");
  auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
  auto expert_tokens_count_or_cumsum_desc = op.GetOutputDesc("expert_tokens_count_or_cumsum");
  auto dynamic_quant_scale_desc = op.GetOutputDesc("dynamic_quant_scale");
  std::vector<int64_t> expanded_x_shape = {3, 5};
  std::vector<int64_t> expanded_idx_shape = {6};
  std::vector<int64_t> expert_tokens_count_or_cumsum_shape = {1};
  std::vector<int64_t> dynamic_quant_scale_shape = {3};
  EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
  EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
  EXPECT_EQ(expert_tokens_count_or_cumsum_desc.GetShape().GetDims(), expert_tokens_count_or_cumsum_shape);
  EXPECT_EQ(dynamic_quant_scale_desc.GetShape().GetDims(), dynamic_quant_scale_shape);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_05) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto idx_tensor_desc = create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({4, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 5);
  op.SetAttr("expert_capacity", 0);
  op.SetAttr("expert_num", 4);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 0);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 1);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto expanded_x_desc = op.GetOutputDesc("expanded_x");
  auto expanded_row_idx_desc = op.GetOutputDesc("expanded_row_idx");
  auto dynamic_quant_scale_desc = op.GetOutputDesc("dynamic_quant_scale");
  std::vector<int64_t> expanded_x_shape = {5, 5};
  std::vector<int64_t> expanded_idx_shape = {6};
  std::vector<int64_t> dynamic_quant_scale_shape = {5};
  EXPECT_EQ(expanded_x_desc.GetShape().GetDims(), expanded_x_shape);
  EXPECT_EQ(expanded_row_idx_desc.GetShape().GetDims(), expanded_idx_shape);
  EXPECT_EQ(dynamic_quant_scale_desc.GetShape().GetDims(), dynamic_quant_scale_shape);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_06) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expert_idx_tensor_desc =
      create_desc_shape_range({3, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 15);
  op.SetAttr("expert_capacity", 6);
  op.SetAttr("expert_num", 4);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 0);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 1);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_07) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expert_idx_tensor_desc =
      create_desc_shape_range({2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 15);
  op.SetAttr("expert_capacity", 6);
  op.SetAttr("expert_num", 4);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 0);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 1);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_08) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expert_idx_tensor_desc =
      create_desc_shape_range({2, 5}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", -1);
  op.SetAttr("expert_capacity", 6);
  op.SetAttr("expert_num", 4);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 0);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 1);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_09) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expert_idx_tensor_desc =
      create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 5);
  op.SetAttr("expert_capacity", -1);
  op.SetAttr("expert_num", 4);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 0);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 1);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_10) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expert_idx_tensor_desc =
      create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 0);
  op.SetAttr("expert_capacity", 0);
  op.SetAttr("expert_num", -1);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 0);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 1);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_11) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expert_idx_tensor_desc =
      create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 0);
  op.SetAttr("expert_capacity", 2);
  op.SetAttr("expert_num", 5);
  op.SetAttr("drop_pad_mode", 99);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 0);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 1);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_12) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expert_idx_tensor_desc =
      create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 0);
  op.SetAttr("expert_capacity", 0);
  op.SetAttr("expert_num", 0);
  op.SetAttr("drop_pad_mode", 1);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 0);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 1);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_13) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expert_idx_tensor_desc =
      create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 6);
  op.SetAttr("expert_capacity", 0);
  op.SetAttr("expert_num", 0);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 1);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 1);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_14) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expert_idx_tensor_desc =
      create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({1, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 6);
  op.SetAttr("expert_capacity", 0);
  op.SetAttr("expert_num", 5);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 1);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 99);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_15) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expert_idx_tensor_desc =
      create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({5, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 6);
  op.SetAttr("expert_capacity", 0);
  op.SetAttr("expert_num", 5);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 1);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 0);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_16) {
  ge::op::MoeInitRoutingQuantV2 op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto x_tensor_desc = create_desc_shape_range({2, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expert_idx_tensor_desc =
      create_desc_shape_range({2, 3}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto scale_tensor_desc =
      create_desc_shape_range({5}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto offset_tensor_desc =
      create_desc_shape_range({1}, ge::DT_FLOAT, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", x_tensor_desc);
  op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);
  op.UpdateInputDesc("scale", scale_tensor_desc);
  op.UpdateInputDesc("offset", offset_tensor_desc);
  op.SetAttr("active_num", 6);
  op.SetAttr("expert_capacity", 0);
  op.SetAttr("expert_num", 5);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("expert_tokens_count_or_cumsum_flag", 1);
  op.SetAttr("expert_tokens_before_capacity_flag", false);
  op.SetAttr("quant_mode", 0);
  Runtime2TestParam param{{"active_num", "expert_capacity", "expert_num", "drop_pad_mode",
                           "expert_tokens_count_or_cumsum_flag", "expert_tokens_before_capacity_flag", "quant_mode"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_datatype_01) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingQuantV2"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingQuantV2")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_FLOAT;
    ge::DataType output_ref = ge::DT_INT8;
    ge::DataType rstd_ref = ge::DT_INT32;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(4, 5)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    EXPECT_EQ(context->GetOutputDataType(3), rstd_ref);
    EXPECT_EQ(context->GetOutputDataType(4), input_ref);
  }
}