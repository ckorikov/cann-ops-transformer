/**
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

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "op_proto_test_util.h" // NOLINT
#include "fusion_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class DequantRopeQuantKvcache : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "DequantRopeQuantKvcache SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DequantRopeQuantKvcache TearDown" << std::endl;
    }
};

TEST_F(DequantRopeQuantKvcache, DequantRopeQuantKvcache_infershape_case_0)
{
    ge::op::DequantRopeQuantKvcache op;
    op.UpdateInputDesc("x", create_desc({4, 1, 1280}, ge::DT_FLOAT16));
    op.UpdateInputDesc("cos", create_desc({4, 1, 1, 128}, ge::DT_FLOAT16));
    op.UpdateInputDesc("sin", create_desc({4, 1, 1, 128}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scale_k", create_desc({128}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale_v", create_desc({128}, ge::DT_INT32));
    op.UpdateInputDesc("k_cache", create_desc({4, 2048, 1, 128}, ge::DT_INT8));
    op.UpdateInputDesc("v_cache", create_desc({4, 2048, 1, 128}, ge::DT_INT8));
    op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));

    std::vector<int64_t> size_splits;
    size_splits.push_back(1024);
    size_splits.push_back(128);
    size_splits.push_back(128);
    op.SetAttr("size_splits", (size_splits));
    op.SetAttr("quant_mode", "static");
    op.SetAttr("layout", "BSND");
    op.SetAttr("kv_output", false);
    Runtime2TestParam param{{"size_splits", "quant_mode", "layout", "kv_output"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
}

TEST_F(DequantRopeQuantKvcache, DequantRopeQuantKvcache_infershape_case_1)
{
    ge::op::DequantRopeQuantKvcache op;
    op.UpdateInputDesc("x", create_desc({4, 1280}, ge::DT_FLOAT16));
    op.UpdateInputDesc("cos", create_desc({4, 128}, ge::DT_FLOAT16));
    op.UpdateInputDesc("sin", create_desc({4, 128}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scale_k", create_desc({128}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale_v", create_desc({128}, ge::DT_INT32));
    op.UpdateInputDesc("k_cache", create_desc({4, 2048, 1, 128}, ge::DT_INT8));
    op.UpdateInputDesc("v_cache", create_desc({4, 2048, 1, 128}, ge::DT_INT8));
    op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));

    std::vector<int64_t> size_splits;
    size_splits.push_back(1024);
    size_splits.push_back(128);
    size_splits.push_back(128);
    op.SetAttr("size_splits", (size_splits));
    op.SetAttr("quant_mode", "static");
    op.SetAttr("layout", "BSND");
    op.SetAttr("kv_output", true);
    Runtime2TestParam param{{"size_splits", "quant_mode", "layout", "kv_output"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
}
