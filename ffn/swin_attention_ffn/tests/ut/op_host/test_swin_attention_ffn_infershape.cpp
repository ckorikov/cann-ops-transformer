/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
