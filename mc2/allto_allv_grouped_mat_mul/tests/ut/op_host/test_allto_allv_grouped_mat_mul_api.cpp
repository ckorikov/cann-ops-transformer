#include <float.h>

#include <array>
#include <vector>

#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "op_host/op_api/aclnn_allto_allv_grouped_mat_mul.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_allto_allv_grouped_mat_mul_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_allto_allv_grouped_mat_mul_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_allto_allv_grouped_mat_mul_test TearDown" << endl; }
};

TEST_F(l2_allto_allv_grouped_mat_mul_test, test) {
  TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto sendCounts = nullptr;
  auto recvCounts = nullptr;   
  bool transGmmWeight = false;
  bool transMmWeight = false;
  bool permuteOutFlag = false;
  int64_t epWorldSize = 8;
  TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul, INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr,
                      "test_allto_allv_grouped_mat_mul_ep_group", epWorldSize, sendCounts, recvCounts, 
					  transGmmWeight, transMmWeight, permuteOutFlag),
                      OUTPUT(gmmY_desc, nullptr, nullptr));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// gmmx null
TEST_F(l2_allto_allv_grouped_mat_mul_test, test_gmmx_null) {
	TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto sendCounts = nullptr;
	auto recvCounts = nullptr;   
	bool transGmmWeight = false;
	bool transMmWeight = false;
	bool permuteOutFlag = false;
	int64_t epWorldSize = 8;
	TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul, INPUT(nullptr, gmmWeight, nullptr, nullptr, nullptr, nullptr,
						"test_allto_allv_grouped_mat_mul_ep_group", epWorldSize, sendCounts, recvCounts, 
						transGmmWeight, transMmWeight, permuteOutFlag),
						OUTPUT(gmmY_desc, nullptr, nullptr));     
	uint64_t workspace_size = 0;
	aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  }

// gmmWeight null
TEST_F(l2_allto_allv_grouped_mat_mul_test, test_gmmWeight_null) {
	TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto sendCounts = nullptr;
	auto recvCounts = nullptr;   
	bool transGmmWeight = false;
	bool transMmWeight = false;
	bool permuteOutFlag = false;
	int64_t epWorldSize = 8;
	TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul, INPUT(gmmX, nullptr, nullptr, nullptr, nullptr, nullptr,
						"test_allto_allv_grouped_mat_mul_ep_group", epWorldSize, sendCounts, recvCounts, 
						transGmmWeight, transMmWeight, permuteOutFlag),
						OUTPUT(gmmY_desc, nullptr, nullptr));     
	uint64_t workspace_size = 0;
	aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  }

// nullptr, nullptr null
TEST_F(l2_allto_allv_grouped_mat_mul_test, test_expertTokenNum_null) {
	TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto sendCounts = nullptr;
	auto recvCounts = nullptr;   
	bool transGmmWeight = false;
	bool transMmWeight = false;	
	bool permuteOutFlag = false;
	int64_t epWorldSize = 8;
	TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul, INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr,
						"test_allto_allv_grouped_mat_mul_ep_group", epWorldSize, sendCounts, recvCounts, 
						transGmmWeight, transMmWeight, permuteOutFlag),
						OUTPUT(gmmY_desc, nullptr, nullptr));     
	uint64_t workspace_size = 0;
	aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  }

// gmmY null 
TEST_F(l2_allto_allv_grouped_mat_mul_test, test_gmmY_null) {
	TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto sendCounts = nullptr;
	auto recvCounts = nullptr;   
	bool transGmmWeight = false;
	bool transMmWeight = false;
	bool permuteOutFlag = false;
	int64_t epWorldSize = 8;
	auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul, INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr,
						"test_allto_allv_grouped_mat_mul_ep_group", epWorldSize, sendCounts, recvCounts, 
						transGmmWeight, transMmWeight, permuteOutFlag),
						OUTPUT(nullptr, nullptr, nullptr));     
	uint64_t workspace_size = 0;
	aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// group ep null
TEST_F(l2_allto_allv_grouped_mat_mul_test, test_groupEp_null) {
	TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);	
	auto sendCounts = nullptr;
	auto recvCounts = nullptr;   
	bool transGmmWeight = false;
	bool transMmWeight = false;
	bool permuteOutFlag = false;
	int64_t epWorldSize = 8;
	TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul, INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr,
						nullptr, epWorldSize, sendCounts, recvCounts, 
						transGmmWeight, transMmWeight, permuteOutFlag),
						OUTPUT(gmmY_desc, nullptr, nullptr));  
	uint64_t workspace_size = 0;
	aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
  
// group ep invalid
TEST_F(l2_allto_allv_grouped_mat_mul_test, test_groupEp_invalid) {
	TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto sendCounts = nullptr;
	auto recvCounts = nullptr;   
	bool transGmmWeight = false;
	bool transMmWeight = false;
	bool permuteOutFlag = false;
	int64_t epWorldSize = 8;
	TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul, INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr,
						"test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group_"
						"test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group_"
						"test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group_"
						"test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group",
						epWorldSize, sendCounts, recvCounts, 
						transGmmWeight, transMmWeight, permuteOutFlag),
						OUTPUT(gmmY_desc, nullptr, nullptr));  
	uint64_t workspace_size = 0;
	aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
  
// mmx not_null mmweight null mmy null
TEST_F(l2_allto_allv_grouped_mat_mul_test, test_mmX_invalid) {
	TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX = TensorDesc({1024, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto sendCounts = nullptr;
	auto recvCounts = nullptr;   
	bool transGmmWeight = false;
	bool transMmWeight = false;
	bool permuteOutFlag = false;
	int64_t epWorldSize = 8;
	TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul, INPUT(gmmX, gmmWeight, nullptr, nullptr, mmX, nullptr,
						"test_allto_allv_grouped_mat_mul_ep_group", epWorldSize, sendCounts, recvCounts, 
						transGmmWeight, transMmWeight, permuteOutFlag),
						OUTPUT(gmmY_desc, nullptr, nullptr));     
	uint64_t workspace_size = 0;
	aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_allto_allv_grouped_mat_mul_test, test_permuteOutFlag_invalid) {
	TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
	TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto sendCounts = nullptr;
	auto recvCounts = nullptr;   
	bool transGmmWeight = false;
	bool transMmWeight = false;
	bool permuteOutFlag = true;
	int64_t epWorldSize = 8;
	TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
	auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul, INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr,
						"test_allto_allv_grouped_mat_mul_ep_group", epWorldSize, sendCounts, recvCounts, 
						transGmmWeight, transMmWeight, permuteOutFlag),
						OUTPUT(gmmY_desc, nullptr, nullptr));
	uint64_t workspace_size = 0;
	aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
