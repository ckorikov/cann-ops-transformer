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
#include "matrix_calculation_ops.h"
#include "common/utils/ut_op_common.h"
#include "array_ops.h"
#include "util/util.h"

class SwinTransformerLnQKV : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SwinTransformerLnQKV Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SwinTransformerLnQKV Proto Test TearDown" << std::endl;
  }
};

TEST_F(SwinTransformerLnQKV, swin_transformer_ln_qkv_test_1) {
ge::op::SwinTransformerLnQKV op;
op.UpdateInputDesc("query", create_desc({2, 2048, 64}, ge::DT_FLOAT16));
op.UpdateInputDesc("key", create_desc({2, 2048, 64}, ge::DT_FLOAT16));

op.SetAttr("epsilon", float(0.001));
op.SetAttr("head_dim", 4);
op.SetAttr("head_num", 4);
op.SetAttr("seq_length", 8);
op.SetAttr("shifts", 0);

int64_t ret = op.InferShapeAndType();

auto attention_output_shape_output_desc = op.GetOutputDescByName("query_output");

Runtime2TestParam param{
{"epsilon", "head_dim", "head_num", "seq_length", "shifts"}};
EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
}