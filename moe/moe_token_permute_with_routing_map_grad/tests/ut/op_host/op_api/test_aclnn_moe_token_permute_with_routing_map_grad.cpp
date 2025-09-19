/**
  * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved. reserved.
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
#include "level2/aclnn_moe_token_permute_with_routing_map_grad.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_token_permute_with_routing_map_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_token_permute_with_routing_map_grad_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_token_permute_with_routing_map_grad_test TearDown" << endl;
    }
};

// dtype bf16
// TEST_F(l2_moe_token_permute_with_routing_map_grad_test, Ascend910B2_moe_token_permute_with_routing_map_grad_droppad_false_fp32)
// {
//     auto permutedTokenOutPutGrad = TensorDesc({1024, 7168}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto permutedProbsOutPutGradOptional = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto sortedIndices = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
//     auto routingMapOptional = TensorDesc({512, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto tokensGradOut = TensorDesc({512, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto probsGradOutOptional = TensorDesc({512, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t numExperts = 512;
//     int64_t tokensNum = 512;
//     bool paddedNum = false;
//     auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
//                         INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
//                               routingMapOptional, numExperts, tokensNum, paddedNum),
//                         OUTPUT(tokensGradOut, probsGradOutOptional));
//     uint64_t workspaceSize = 0;
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
// }

TEST_F(l2_moe_token_permute_with_routing_map_grad_test, Ascend910B2_moe_token_permute_with_routing_map_grad_droppad_true_fp32)
{
    auto permutedTokenOutPutGrad = TensorDesc({1024, 7168}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto routingMapOptional = TensorDesc({512, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tokensGradOut = TensorDesc({512, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    auto probsGradOutOptional = TensorDesc({512, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numExperts = 512;
    int64_t tokensNum = 512;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
                              routingMapOptional, numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, probsGradOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}
TEST_F(l2_moe_token_permute_with_routing_map_grad_test, Ascend910B2_moe_token_permute_with_routing_map_grad_droppad_false_fp32)
{
    auto permutedTokenOutPutGrad = TensorDesc({1024, 7168}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto routingMapOptional = TensorDesc({512, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tokensGradOut = TensorDesc({512, 7168}, ACL_FLOAT, ACL_FORMAT_ND);
    auto probsGradOutOptional = TensorDesc({512, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t numExperts = 512;
    int64_t tokensNum = 512;
    bool paddedNum = false;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
                              routingMapOptional, numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, probsGradOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test, Ascend910B2_moe_token_permute_with_routing_map_grad_droppad_true_fp16)
{
    auto permutedTokenOutPutGrad = TensorDesc({1024, 7168}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto routingMapOptional = TensorDesc({512, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tokensGradOut = TensorDesc({512, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOutOptional = TensorDesc({512, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numExperts = 512;
    int64_t tokensNum = 512;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
                              routingMapOptional, numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, probsGradOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}

TEST_F(l2_moe_token_permute_with_routing_map_grad_test, Ascend910B2_moe_token_permute_with_routing_map_grad_empty_tensor)
{
    auto permutedTokenOutPutGrad = TensorDesc({1024, 0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto permutedProbsOutPutGradOptional = TensorDesc({1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto sortedIndices = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto routingMapOptional = TensorDesc({512, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto tokensGradOut = TensorDesc({512, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto probsGradOutOptional = TensorDesc({512, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t numExperts = 512;
    int64_t tokensNum = 512;
    bool paddedNum = true;
    auto ut = OP_API_UT(aclnnMoeTokenPermuteWithRoutingMapGrad,
                        INPUT(permutedTokenOutPutGrad, permutedProbsOutPutGradOptional, sortedIndices,
                              routingMapOptional, numExperts, tokensNum, paddedNum),
                        OUTPUT(tokensGradOut, probsGradOutOptional));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}