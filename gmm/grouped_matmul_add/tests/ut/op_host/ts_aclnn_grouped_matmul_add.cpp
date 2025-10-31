/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ts_aclnn_grouped_matmul_add.cpp
 * \brief GroupedMatmulAdd ACLNN 测试用例.
 */

#include "ts_aclnn_grouped_matmul_add.h"

using gmmAddTestParam::Ts_Aclnn_GroupedMatmulAdd_WithParam_Ascend910_9591;
namespace {

TEST_P(Ts_Aclnn_GroupedMatmulAdd_WithParam_Ascend910_9591, Tc_Tiling_GmmAdd)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_Gmm_Add_Aclnn_David_Case = ::testing::Values(AclnnGroupedMatmulAddCase(
    "AclnnGroupedMatmulAdd_Case0", true, "", /* CaseName, Enable, DebugInfo */
    OpInfo(ControlInfo(true, false),
           ExpectInfo(false,
                        ExpectInfo::kInvalidTilingKey,
                        ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
    AclnnGroupedMatmulAddParam({GenTensor("x", {512, 96}, ge::DataType::DT_FLOAT16),
           GenTensor("weight", {512, 128}, ge::DataType::DT_FLOAT16),
            GenTensor("groupList", {4}, ge::DataType::DT_INT64)
           GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT)}, {128, 256, 300, 512}, true, false, 2, 0), 0
    ));
INSTANTIATE_TEST_SUITE_P(GroupedMatmulAdd, Ts_Aclnn_GroupedMatmulAdd_WithParam_Ascend910_9591, Tc_Gmm_Add_Aclnn_David_Case);
}  // namespace