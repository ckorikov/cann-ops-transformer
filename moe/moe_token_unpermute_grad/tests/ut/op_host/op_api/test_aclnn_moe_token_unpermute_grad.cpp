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

/*!
 * \file test_aclnn_moe_token_unpermute_grad.cpp
 * \brief
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "level2/aclnn_moe_token_unpermute_grad.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_unpermute_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_token_unpermute_grad_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_token_unpermute_grad_test TearDown" << endl;
    }
};

// dtype fp32
TEST_F(l2_moe_token_unpermute_grad_test, Ascend910B2_moe_token_unpermute_grad_fp32)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnMoeTokenUnpermuteGrad,
        INPUT(permuteTokens, unpermutedTokensGrad, SortedIndices, probsOptional, false, (aclIntArray*)nullptr),
        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_moe_token_unpermute_grad_test, Ascend910_9589_moe_token_unpermute_grad_fp32)
{
    auto permuteTokens = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto unpermutedTokensGrad = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto SortedIndices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto probsOptional = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto gradExpandedXOut = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradScalesOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnMoeTokenUnpermuteGrad,
        INPUT(permuteTokens, unpermutedTokensGrad, SortedIndices, probsOptional, false, (aclIntArray*)nullptr),
        OUTPUT(gradExpandedXOut, gradScalesOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
}