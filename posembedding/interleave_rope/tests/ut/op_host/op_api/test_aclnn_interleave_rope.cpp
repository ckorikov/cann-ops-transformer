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

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "aclnn_interleave_rope.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_interleave_rope_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_interleave_rope_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_interleave_rope_test TearDown" << endl;
    }
};

// dtype fp32
TEST_F(l2_interleave_rope_test, Ascend910B2_interleave_rope_fp32)
{
    auto x = TensorDesc({8, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto cos = TensorDesc({8, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 8);
    auto sin = TensorDesc({8, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 8);
    auto out = TensorDesc({8, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInterleaveRope, INPUT(x, cos, sin), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}