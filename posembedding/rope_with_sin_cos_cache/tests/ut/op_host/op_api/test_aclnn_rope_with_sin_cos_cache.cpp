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
#include "gtest/gtest.h"

#include "level2/aclnn_rope_with_sin_cos_cache.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_rope_with_sin_cos_cache_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "rope_with_sin_cos_cache_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "rope_with_sin_cos_cache_test TearDown" << endl;
    }
};

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_1)
{
    auto tensorPositions =
        TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_2)
{
    auto tensorPositions =
        TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64, 50}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64, 51}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64, 51}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_3)
{
    auto tensorPositions =
        TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64, 51}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64, 51}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_4)
{
    auto tensorPositions =
        TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64, 51}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_5)
{
    auto tensorPositions =
        TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_6)
{
    auto tensorPositions =
        TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}