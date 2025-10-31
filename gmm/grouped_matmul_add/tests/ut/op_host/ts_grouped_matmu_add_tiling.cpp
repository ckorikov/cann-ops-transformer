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
 * \file ts_grouped_matmul_add_tiling.cpp
 * \brief GroupedMatmulAdd tiling用例.
 */

#include "ts_grouped_matmul_add.h"
using gmmAddTestParam::Ts_GroupedMatmulAdd_WithParam_Ascend910_9591;
namespace {
TEST_P(Ts_GroupedMatmulAdd_WithParam_Ascend910_9591, Tc_Tiling_GmmAdd)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_GroupedMatmulAdd_Tiling_Case = ::testing::Values(GroupedMatmulAddCase(
    "GroupedMatmulAdd_Case0", true, "", /* CaseName, Enable, DebugInfo */
    OpInfo(ControlInfo(true, false),
           ExpectInfo(true, 10000900009000090001UL, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
    Param({GenTensorList("x", {{512, 96}}, ge::DataType::DT_FLOAT16),
           GenTensorList("weight", {{512, 128}}, ge::DataType::DT_FLOAT16),
           GenTensorList("y", {{4, 96, 128}}, ge::DataType::DT_FLOAT)},
          GenTensor("group_list", {4}, ge::DataType::DT_INT64), {128, 256, 300, 512}, true, false, 2, 0),
    0));

INSTANTIATE_TEST_SUITE_P(GroupedMatmulAdd, Ts_GroupedMatmulAdd_WithParam_Ascend910_9591,
                         Tc_GroupedMatmulAdd_Tiling_Case);


} // namespace