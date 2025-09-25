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
#include "fusion_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class MoeComputeExpertTokensProto : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeComputeExpertTokens SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeComputeExpertTokens TearDown" << std::endl;
    }
};

TEST_F(MoeComputeExpertTokensProto, normal_shape_1)
{
    ge::op::MoeComputeExpertTokens op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 64}};
    auto tensor_desc = create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("sorted_experts", tensor_desc);
    op.SetAttr("num_experts", 6);
    Runtime2TestParam param{{"num_experts"}};
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("total_rows_before_expert");
    std::vector<int64_t> expected_output_shape = {6};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
