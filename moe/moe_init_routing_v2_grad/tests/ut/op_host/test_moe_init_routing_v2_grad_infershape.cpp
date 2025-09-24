/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include <gtest/gtest.h>  // NOLINT
#include <iostream>
#include "op_proto_test_util.h"  // NOLINT
#include "fusion_ops.h"      // NOLINT
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class MoeInitRoutingV2Grad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeInitRoutingV2Grad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeInitRoutingV2Grad TearDown" << std::endl;
  }
};

TEST_F(MoeInitRoutingV2Grad, moe_init_routing_v2_grad_infer_shape_01) {
  ge::op::MoeInitRoutingV2Grad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto grad_expanded_x_tensor_desc =
      create_desc_shape_range({-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expanded_row_idx_tensor_desc =
      create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad_expanded_x", grad_expanded_x_tensor_desc);
  op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);
  op.SetAttr("top_k", 6);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("active_num", 0);
  Runtime2TestParam param{{"top_k", "drop_pad_mode", "active_num"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto grad_x_desc = op.GetOutputDesc("grad_x");
  std::vector<int64_t> grad_x_shape = {-1, -1};
  EXPECT_EQ(grad_x_desc.GetShape().GetDims(), grad_x_shape);
}

TEST_F(MoeInitRoutingV2Grad, moe_init_routing_v2_grad_infer_shape_02) {
  ge::op::MoeInitRoutingV2Grad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto grad_expanded_x_tensor_desc =
      create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expanded_row_idx_tensor_desc =
      create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad_expanded_x", grad_expanded_x_tensor_desc);
  op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);
  op.SetAttr("top_k", 6);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("active_num", 0);
  Runtime2TestParam param{{"top_k", "drop_pad_mode", "active_num"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto grad_x_desc = op.GetOutputDesc("grad_x");
  std::vector<int64_t> grad_x_shape = {-1, -1};
  EXPECT_EQ(grad_x_desc.GetShape().GetDims(), grad_x_shape);
}

TEST_F(MoeInitRoutingV2Grad, moe_init_routing_v2_grad_infer_shape_03) {
  ge::op::MoeInitRoutingV2Grad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto grad_expanded_x_tensor_desc =
      create_desc_shape_range({1024, 512}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expanded_row_idx_tensor_desc =
      create_desc_shape_range({1024}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad_expanded_x", grad_expanded_x_tensor_desc);
  op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);
  op.SetAttr("top_k", 64);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("active_num", 0);
  Runtime2TestParam param{{"top_k", "drop_pad_mode", "active_num"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto grad_x_desc = op.GetOutputDesc("grad_x");
  std::vector<int64_t> grad_x_shape = {16, 512};
  EXPECT_EQ(grad_x_desc.GetShape().GetDims(), grad_x_shape);
}

TEST_F(MoeInitRoutingV2Grad, moe_init_routing_v2_grad_infer_shape_04) {
  ge::op::MoeInitRoutingV2Grad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto grad_expanded_x_tensor_desc =
      create_desc_shape_range({1024, 512}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expanded_row_idx_tensor_desc =
      create_desc_shape_range({1024}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad_expanded_x", grad_expanded_x_tensor_desc);
  op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);
  op.SetAttr("top_k", 0);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("active_num", 0);
  Runtime2TestParam param{{"top_k", "drop_pad_mode", "active_num"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2Grad, moe_init_routing_v2_grad_infer_shape_05) {
  ge::op::MoeInitRoutingV2Grad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto grad_expanded_x_tensor_desc =
      create_desc_shape_range({1024, 512}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expanded_row_idx_tensor_desc =
      create_desc_shape_range({1024}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad_expanded_x", grad_expanded_x_tensor_desc);
  op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);
  op.SetAttr("top_k", 64);
  op.SetAttr("drop_pad_mode", 2);
  op.SetAttr("active_num", 0);
  Runtime2TestParam param{{"top_k", "drop_pad_mode", "active_num"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2Grad, moe_init_routing_v2_grad_infer_shape_06) {
  ge::op::MoeInitRoutingV2Grad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto grad_expanded_x_tensor_desc =
      create_desc_shape_range({1024, 512}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expanded_row_idx_tensor_desc =
      create_desc_shape_range({1024}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad_expanded_x", grad_expanded_x_tensor_desc);
  op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);
  op.SetAttr("top_k", 64);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("active_num", -1);
  Runtime2TestParam param{{"top_k", "drop_pad_mode", "active_num"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2Grad, moe_init_routing_v2_grad_infer_shape_07) {
  ge::op::MoeInitRoutingV2Grad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto grad_expanded_x_tensor_desc =
      create_desc_shape_range({1024, 512}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expanded_row_idx_tensor_desc =
      create_desc_shape_range({1024, 512}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad_expanded_x", grad_expanded_x_tensor_desc);
  op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);
  op.SetAttr("top_k", 64);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("active_num", -1);
  Runtime2TestParam param{{"top_k", "drop_pad_mode", "active_num"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2Grad, moe_init_routing_v2_grad_infer_shape_08) {
  ge::op::MoeInitRoutingV2Grad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto grad_expanded_x_tensor_desc =
      create_desc_shape_range({1024, 512}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expanded_row_idx_tensor_desc =
      create_desc_shape_range({1024}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad_expanded_x", grad_expanded_x_tensor_desc);
  op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);
  op.SetAttr("top_k", 64);
  op.SetAttr("drop_pad_mode", 2);
  op.SetAttr("active_num", -1);
  Runtime2TestParam param{{"top_k", "drop_pad_mode", "active_num"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2Grad, moe_init_routing_v2_grad_infer_shape_09) {
  ge::op::MoeInitRoutingV2Grad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto grad_expanded_x_tensor_desc =
      create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expanded_row_idx_tensor_desc =
      create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad_expanded_x", grad_expanded_x_tensor_desc);
  op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);
  op.SetAttr("top_k", 6);
  op.SetAttr("drop_pad_mode", 1);
  op.SetAttr("active_num", 0);
  Runtime2TestParam param{{"top_k", "drop_pad_mode", "active_num"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto grad_x_desc = op.GetOutputDesc("grad_x");
  std::vector<int64_t> grad_x_shape = {-1, -1};
  EXPECT_EQ(grad_x_desc.GetShape().GetDims(), grad_x_shape);
}

TEST_F(MoeInitRoutingV2Grad, moe_init_routing_v2_grad_infer_shape_10) {
  ge::op::MoeInitRoutingV2Grad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto grad_expanded_x_tensor_desc =
      create_desc_shape_range({-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expanded_row_idx_tensor_desc =
      create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad_expanded_x", grad_expanded_x_tensor_desc);
  op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);
  op.SetAttr("top_k", 6);
  op.SetAttr("drop_pad_mode", 1);
  op.SetAttr("active_num", 0);
  Runtime2TestParam param{{"top_k", "drop_pad_mode", "active_num"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2Grad, moe_init_routing_v2_grad_infer_shape_11) {
  ge::op::MoeInitRoutingV2Grad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
  auto grad_expanded_x_tensor_desc =
      create_desc_shape_range({1024, 512}, ge::DT_FLOAT16, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  auto expanded_row_idx_tensor_desc =
      create_desc_shape_range({1024, 512}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grad_expanded_x", grad_expanded_x_tensor_desc);
  op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);
  op.SetAttr("top_k", 64);
  op.SetAttr("drop_pad_mode", 0);
  op.SetAttr("active_num", 0);
  Runtime2TestParam param{{"top_k", "drop_pad_mode", "active_num"}};
  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}