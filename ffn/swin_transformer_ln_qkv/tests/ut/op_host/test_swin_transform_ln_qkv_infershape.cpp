/**
Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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