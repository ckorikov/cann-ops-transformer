/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "array_ops.h"
#include "fusion_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/attr_utils.h"
#include "common/utils/ut_op_common.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "util/util.h"
#include "op_proto_test_util.h"

class SwinTransformerLnQkvQuantTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SwinTransformerLnQkvQuantInferShapeTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SwinTransformerLnQkvQuantInferShapeTest TearDown" << std::endl;
  }
};

TEST_F(SwinTransformerLnQkvQuantTest, swin_transformer_ln_qkv_quant_infer_shape_fp16) {
  ge::op::SwinTransformerLnQkvQuant op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 8000}, {1, 8000}, {1, 8000}};
  auto tensor_desc = create_desc_shape_range({1, 6272, 96},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1, 6272, 96},
                                             ge::FORMAT_ND, shape_range);
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{1, 8000}};
  auto tensor_desc1 = create_desc_shape_range({96},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {96},
                                             ge::FORMAT_ND, shape_range1);
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{1, 8000}};
  auto tensor_desc2 = create_desc_shape_range({96},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {96},
                                             ge::FORMAT_ND, shape_range2);
  std::vector<std::pair<int64_t,int64_t>> shape_range3 = {{1, 8000},{1, 8000}};
  auto tensor_desc3 = create_desc_shape_range({288, 96},
                                             ge::DT_INT8, ge::FORMAT_ND,
                                             {288, 96},
                                             ge::FORMAT_ND, shape_range3);
  std::vector<std::pair<int64_t,int64_t>> shape_range4 = {{1, 8000}};
  auto tensor_desc4 = create_desc_shape_range({288},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {288},
                                             ge::FORMAT_ND, shape_range4);
  std::vector<std::pair<int64_t,int64_t>> shape_range5 = {{1, 8000}};
  auto tensor_desc5 = create_desc_shape_range({96},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {96},
                                             ge::FORMAT_ND, shape_range5);
  std::vector<std::pair<int64_t,int64_t>> shape_range6 = {{1, 8000}};
  auto tensor_desc6 = create_desc_shape_range({96},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {96},
                                             ge::FORMAT_ND, shape_range6);
  std::vector<std::pair<int64_t,int64_t>> shape_range7 = {{1, 8000}};
  auto tensor_desc7 = create_desc_shape_range({288},
                                             ge::DT_UINT64, ge::FORMAT_ND,
                                             {288},
                                             ge::FORMAT_ND, shape_range7);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("gamma", tensor_desc1);
  op.UpdateInputDesc("beta", tensor_desc2);
  op.UpdateInputDesc("weight", tensor_desc3);
  op.UpdateInputDesc("bias", tensor_desc4);
  op.UpdateInputDesc("quant_scale", tensor_desc5);
  op.UpdateInputDesc("quant_offset", tensor_desc6);
  op.UpdateInputDesc("dequant_scale", tensor_desc7);
  op.SetAttr("head_num", 3);
  op.SetAttr("seq_length", 32);
  op.SetAttr("epsilon", float(0.00001));
  op.SetAttr("ori_height", 56);
  op.SetAttr("ori_weight", 112);
  op.SetAttr("h_win_size", 7);
  op.SetAttr("w_win_size", 7);
  op.SetAttr("weight_transpose", true);
  auto ret = op.InferShapeAndType();

  Runtime2TestParam param{
  {"head_num", "seq_length", "epsilon", "ori_height", "ori_weight", "h_win_size", "w_win_size", "weight_transpose"}};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
}