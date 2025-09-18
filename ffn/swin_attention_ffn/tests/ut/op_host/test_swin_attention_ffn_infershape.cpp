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

class swin_attention_ffn : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "SwinAttentionFFN Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SwinAttentionFFN Proto Test TearDown" << std::endl;
    }
};

TEST_F(swin_attention_ffn, swin_attention_ffn_test_1) {
    ge::op::SwinAttentionFFN op;
    
    // update op input
    op.UpdateInputDesc("x1", create_desc({4096, 64, 128}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({128, 128}, ge::DT_FLOAT16));
    op.UpdateInputDesc("bias", create_desc({128}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x3", create_desc({4096, 64, 128}, ge::DT_FLOAT16));
    op.SetAttr("shifts", {4, 4});

    // call InferShapeAndType function
    auto ret = op.InferShapeAndType();

    auto y_shape_output_desc = op.GetOutputDescByName("y");

    Runtime2TestParam param{{"shifts"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
}
