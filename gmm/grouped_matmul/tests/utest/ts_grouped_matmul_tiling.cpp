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
 * \file ts_grouped_matmul_tiling.cpp
 * \brief GroupedMatmul tiling用例.
 */

 #include "ts_grouped_matmul.h"

using gmmTestParam::Ts_GroupedMatmul_WithParam_Ascend310P3;
using gmmTestParam::Ts_GroupedMatmul_WithParam_Ascend910_9591;
using gmmTestParam::Ts_GroupedMatmul_WithParam_Ascend910B2;
using gmmTestParam::Ts_GroupedMatmul_WithParam_Ascend910B3;
namespace {
TEST_P(Ts_GroupedMatmul_WithParam_Ascend910B2, Tc_Kernel_910b2_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

TEST_P(Ts_GroupedMatmul_WithParam_Ascend910B3, Tc_Tiling_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

TEST_P(Ts_GroupedMatmul_WithParam_Ascend310P3, Tc_Tiling310_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_GroupedMatmul_Tiling_Case = ::testing::Values(
    GroupedMatmulCase(
        "GroupedMatmul_Case0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{512, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase(
        "GroupedMatmul_Case1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 300}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{300, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 0, -1, false, false, 1, 1, 0),
        0),
    GroupedMatmulCase(
        "GroupedMatmul_Case2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 400}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{400, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}, {256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {128, 128}, 0, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase(
        "GroupedMatmul_Case3", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 300}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{300, 128}, {300, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {128, 128}, 3, -1, false, false, 2, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor split m */
        "GroupedMatmul_Case4", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 512, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{4, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor + transpose w */
        "GroupedMatmul_Case5", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 128, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{4, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor + transpose x */
        "GroupedMatmul_Case6", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 1,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{512, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 512, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{4, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, false, true, 0, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant */
        "GroupedMatmul_Case7", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{16, 5}, {16, 5}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{5, 10}, {5, 10}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{10}, {10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{10}, {10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{10}, {10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}, {16, 10}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, -1, 0, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case8", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{360, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{16, 1024, 8191}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 8191}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{360, 8191}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {16}, ge::DataType::DT_INT64),
              {40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case9", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 17,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{360, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{16, 256, 64, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 8192}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{360, 8192}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {16}, ge::DataType::DT_INT64),
              {20, 20, 40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case10", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 17,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{360, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{32, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{32, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{360, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {32}, ge::DataType::DT_INT64),
              {11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 12, 11, 12, 11, 12,
               11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 12, 11, 12, 11, 12}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case11", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 17,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{4, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {30, 30, 30, 30}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant int32 weight NZ */
        "GroupedMatmul_Case12", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 13, ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{0}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {120}, 2, 2, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case13", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 17, ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {120}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case14", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 4, 8)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {120}, 2, 1, false, false, 0, 1, 0, {200}),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case15", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 4, 8)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {120}, 2, 1, false, false, 0, 1, 0, {200}),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case16", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0, 20)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{9, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{16, 2048, 32768}}, ge::DataType::DT_INT8, ge::FORMAT_ND),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 328768}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{9, 32768}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {9}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {120}, 2, 1, false, false, 0, 1, 0, {9}),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case17", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 4, 8)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 7680}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 7680,1024}}, ge::DataType::DT_INT8, ge::FORMAT_ND),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {120}, 2, 1, false, false, 0, 1, 0, {256}),
        0),
    GroupedMatmulCase(
            "GroupedMatmul_Case18", true, "",
            OpInfo(ControlInfo(true, false), \
              ExpectInfo(true, 0, 20)),
            Param({GenTensorList("x", {{9, 7168}, {9, 7168}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{7168, 2048}, {7168, 2048}}, ge::DataType::DT_INT8),
                GenTensorList("bias", {{2048}, {2048}}, ge::DataType::DT_INT32),
                GenTensorList("scale", {{}}, ge::DataType::DT_INT32),
                GenTensorList("offset", {{}}, ge::DataType::DT_INT32),
                GenTensorList("antiquant_scale", {{2048}, {2048}}, ge::DataType::DT_INT32),
                GenTensorList("antiquant_offset", {{2048}, {2048}}, ge::DataType::DT_INT32),
                GenTensorList("y", {{9, 2048}, {9, 2048}}, ge::DataType::DT_INT32)},
            GenTensor("per_token_scale", {}, ge::DataType::DT_INT32),
            GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
            {}, 0, -1, false, false, -1, 0, 0, {9}),
            0),
    GroupedMatmulCase(
              "GroupedMatmul_Case_a4w4_pergroup", true, "", /* CaseName, Enable, DebugInfo */
              OpInfo(ControlInfo(true, false),
                     ExpectInfo(true, 4, 20)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
              Param({GenTensorList("x", {{64, 256}}, ge::DataType::DT_INT4),
                     GenTensorList("weight", {{1, 256, 128}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                     GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("scale", {{1,1,128}}, ge::DataType::DT_UINT64),
                     GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                     GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("y", {{64, 128}}, ge::DataType::DT_FLOAT16)},
                     GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                     GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                    {64}, 3, -1, false, false, 0, 1, 0),
              0),
     GroupedMatmulCase(
              "GroupedMatmul_Case_a4w4_perchannel", true, "", /* CaseName, Enable, DebugInfo */
              OpInfo(ControlInfo(true, false),
                     ExpectInfo(true, 4, 20)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
              Param({GenTensorList("x", {{64, 256}}, ge::DataType::DT_INT4),
                     GenTensorList("weight", {{1, 256, 128}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                     GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("scale", {{1, 128}}, ge::DataType::DT_UINT64),
                     GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                     GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("y", {{64, 128}}, ge::DataType::DT_FLOAT16)},
                     GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                     GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                     {64}, 3, -1, false, false, 0, 1, 0),
              0),
    GroupedMatmulCase(
        "GroupedMatmul_Case_FullLoad", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{25600, 384}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 384, 2560}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{25600, 2560}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {6400, 6400, 6400, 6400}, 3, -1, false, false, 0, 1, 0),
        0)
);

INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_GroupedMatmul_WithParam_Ascend910B3, Tc_GroupedMatmul_Tiling_Case);

const auto Tc_GroupedMatmul_Kernel910B2_Case = ::testing::Values(
    GroupedMatmulCase( /* a8w8 preTiling */
        "GroupedMatmul_Pretiling_tiling_Case_0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{576, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 7168, 4096}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 4096}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{576, 4096}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {576}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 preTiling */
        "GroupedMatmul_Pretiling_tiling_Case_1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{228, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 7168, 4096}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 4096}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{228, 4096}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {228}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 preTiling */
        "GroupedMatmul_Pretiling_tiling_Case_2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{228, 2048}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 2048, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 7168}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{228, 7168}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {228}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 13, 24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 7168, 4096}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {0, 540}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 14, 24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{1080, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 4096, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{1, 1080, 4096}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {540, 540}, 3, 1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_4", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 15, 24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 7168, 4096}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {{0, 540, 1, 0}}, 3, 1, false, false, 0, 2, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_5", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 16, 24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 4096, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {{0, 0, 1, 540}}, 3, 1, true, false, 0, 2, 0),
        0),
       GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_8", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 17, 24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 7168, 4096}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {540}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {0, 540}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_9", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 18, 24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{1080, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 4096, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{1, 1080, 4096}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {1080}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {540, 540}, 3, 1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_12", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 19, 24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 7168, 4096}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {{0, 540, 1, 0}}, 3, 1, false, false, 0, 2, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_13", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20, 24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 4096, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {540}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {{0, 0, 1, 540}}, 3, 1, true, false, 0, 2, 0),
        0)
);

INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_GroupedMatmul_WithParam_Ascend910B2, Tc_GroupedMatmul_Kernel910B2_Case);


const auto Tc_GroupedMatmul_Tiling310_Case = ::testing::Values(
       GroupedMatmulCase(
           "GroupedMatmul_Case0", true, "", /* CaseName, Enable, DebugInfo */
           OpInfo(ControlInfo(true, false),
                  ExpectInfo(true, 0, 8)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
           Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("weight", {{512, 128}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
                 GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                 {64, 64, 64, 64}, 0, -1, false, false, -1, 1, 0),
           0));

INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_GroupedMatmul_WithParam_Ascend310P3, Tc_GroupedMatmul_Tiling310_Case);

TEST_P(Ts_GroupedMatmul_WithParam_Ascend910_9591, Tc_Tiling_Gmm_david)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_Gmm_Tiling_Case_David = ::testing::Values(
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020003000012020,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 128, 512}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{4, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{4, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 64, 64},  // groupListData
              2,                                                                         // splitItem
              -1,                                                                        // dType
              true,                                                                      // transposeWeight
              false,                                                                     // transposeX
              0,                                                                         // groupType
              1,                                                                         // groupListType
              0),                                                                        // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020003000012000,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{4, 128, 512}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{4, 128}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 128, 256},  // groupListData
              2,                                                                           // splitItem
              -1,                                                                          // dType
              true,                                                                        // transposeWeight
              false,                                                                       // transposeX
              0,                                                                           // groupType
              0,                                                                           // groupListType
              0),                                                                          // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020004000002001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 1024, 64, 16, 32}}, ge::DataType::DT_INT4,
                             ge::FORMAT_FRACTAL_NZ),  // (N1, K1, K0, N0)
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1, 32768}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{4, 8, 32768}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 32768}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 1}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 128, 256},  // groupListData
              2,                                                                           // splitItem
              -1,                                                                          // dType
              false,                                                                       // transposeWeight
              false,                                                                       // transposeX
              0,                                                                           // groupType
              0,                                                                           // groupListType
              0),                                                                          // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case3", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020004000002001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 640, 64, 16, 32}}, ge::DataType::DT_INT4,
                             ge::FORMAT_FRACTAL_NZ),  // (N1, K1, K0, N0)
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1, 20480}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{4, 8, 20480}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 20480}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 1}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 0, 64},  // groupListData
              2,                                                                        // splitItem
              -1,                                                                       // dType
              false,                                                                    // transposeWeight
              false,                                                                    // transposeX
              0,                                                                        // groupType
              1,                                                                        // groupListType
              0),                                                                       // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case4", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020004000002001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 641, 64, 16, 32}}, ge::DataType::DT_INT4,
                             ge::FORMAT_FRACTAL_NZ),  // (N1, K1, K0, N0)
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1, 20512}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{4, 8, 20512}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 20512}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 1}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 128, 256},  // groupListData
              2,                                                                           // splitItem
              -1,                                                                          // dType
              false,                                                                       // transposeWeight
              false,                                                                       // transposeX
              0,                                                                           // groupType
              0,                                                                           // groupListType
              0),                                                                          // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case5", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020004000002001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 160, 64, 16, 32}}, ge::DataType::DT_INT4,
                             ge::FORMAT_FRACTAL_NZ),  // (N1, K1, K0, N0)
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1, 5120}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{4, 8, 5120}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 5120}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 1}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 128, 256},  // groupListData
              2,                                                                           // splitItem
              -1,                                                                          // dType
              false,                                                                       // transposeWeight
              false,                                                                       // transposeX
              0,                                                                           // groupType
              0,                                                                           // groupListType
              0),                                                                          // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case6", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020004000002001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 130, 64, 16, 32}}, ge::DataType::DT_INT4,
                             ge::FORMAT_FRACTAL_NZ),  // (N1, K1, K0, N0)
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1, 4160}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{4, 8, 4160}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 4160}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 1}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 128, 256},  // groupListData
              2,                                                                           // splitItem
              -1,                                                                          // dType
              false,                                                                       // transposeWeight
              false,                                                                       // transposeX
              0,                                                                           // groupType
              0,                                                                           // groupListType
              0),                                                                          // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case7", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020004000002001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 64, 64, 16, 32}}, ge::DataType::DT_INT4,
                             ge::FORMAT_FRACTAL_NZ),  // (N1, K1, K0, N0)
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1, 2048}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{4, 8, 2048}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 2048}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 1}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 128, 256},  // groupListData
              2,                                                                           // splitItem
              -1,                                                                          // dType
              false,                                                                       // transposeWeight
              false,                                                                       // transposeX
              0,                                                                           // groupType
              0,                                                                           // groupListType
              0),                                                                          // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case8", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020004000002001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 65, 64, 16, 32}}, ge::DataType::DT_INT4,
                             ge::FORMAT_FRACTAL_NZ),  // (N1, K1, K0, N0)
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1, 2080}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{4, 8, 2080}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 2080}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 1}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 128, 256},  // groupListData
              2,                                                                           // splitItem
              -1,                                                                          // dType
              false,                                                                       // transposeWeight
              false,                                                                       // transposeX
              0,                                                                           // groupType
              0,                                                                           // groupListType
              0),                                                                          // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case9", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020004000002001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 160, 64, 16, 32}}, ge::DataType::DT_INT4,
                             ge::FORMAT_FRACTAL_NZ),  // (N1, K1, K0, N0)
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1, 5120}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{4, 4, 5120}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 5120}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 1}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 128, 256},  // groupListData
              2,                                                                           // splitItem
              -1,                                                                          // dType
              false,                                                                       // transposeWeight
              false,                                                                       // transposeX
              0,                                                                           // groupType
              0,                                                                           // groupListType
              0),                                                                          // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case10", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020004000002001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 160, 64, 16, 32}}, ge::DataType::DT_INT4,
                             ge::FORMAT_FRACTAL_NZ),  // (N1, K1, K0, N0)
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1, 5120}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{4, 2, 5120}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 5120}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 1}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 128, 256},  // groupListData
              2,                                                                           // splitItem
              -1,                                                                          // dType
              false,                                                                       // transposeWeight
              false,                                                                       // transposeX
              0,                                                                           // groupType
              0,                                                                           // groupListType
              0),                                                                          // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case11", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020003000004001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{4, 1280, 64, 16, 16}}, ge::DataType::DT_FLOAT4_E2M1,
                             ge::FORMAT_FRACTAL_NZ),  // (N1, K1, K0, N0)
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{4, 32, 20480}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 20480}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 0, 64},  // groupListData
              2,                                                                        // splitItem
              -1,                                                                       // dType
              false,                                                                    // transposeWeight
              false,                                                                    // transposeX
              0,                                                                        // groupType
              1,                                                                        // groupListType
              0),                                                                       // actType
        0),
    GroupedMatmulCase(
        "GroupedWeightQuantBatchMatmul_Case12", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000020003000014001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 1024}}, ge::DataType::DT_FLOAT8_E4M3FN),
               GenTensorList("weight", {{4, 32, 20, 16, 32}}, ge::DataType::DT_FLOAT4_E2M1,
                             ge::FORMAT_FRACTAL_NZ),  // (K1, N1, N0, K0)
               GenTensorList("bias", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{4, 320, 32}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 320}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 32}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 64, 0, 64},  // groupListData
              2,                                                                        // splitItem
              -1,                                                                       // dType
              true,                                                                     // transposeWeight
              false,                                                                    // transposeX
              0,                                                                        // groupType
              1,                                                                        // groupListType
              0),                                                                       // actType
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp8_typem_false_true", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000001, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("weight", {{4, 128, 512}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 128, 8, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 8, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {4}, ge::DataType::DT_INT64), {64, 64, 64, 64}, 2, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp8_typem_false_false", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000000, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("weight", {{4, 512, 128}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 8, 128, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 8, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {4}, ge::DataType::DT_INT64), {64, 64, 64, 64}, 2, -1, false, false, 0, 1, 0),
        0),
   GroupedMatmulCase(
        "GroupedQuantMM_mxfp8_typem_true_false_error", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 20000000001, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{512, 256}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("weight", {{4, 512, 128}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 8, 128, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {256, 8, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {4}, ge::DataType::DT_INT64), {64, 64, 64, 64}, 2, -1, false, true, 0, 1, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp8_typek_true_false_0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000010, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{960, 2068}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("weight", {{960, 1014}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{19, 1014, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{1, 2068, 1014}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {19, 2068, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {4}, ge::DataType::DT_INT64), {0,}, 2, -1, false, true, 2, 1, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp8_typek_true_false_1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000010, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{8192, 7168}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("weight", {{8192, 2048}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{132, 2048, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4, 7168, 2048}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {132, 7168, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {4}, ge::DataType::DT_INT64), {64, 0, 7168, 64}, 2, -1, false, true, 2, 1, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_fp8_perTensor_scale_uint64", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000001, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{2, 448}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("weight", {{2, 2043, 448}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{2, 1}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 2043}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {1, 2}, 2, -1, true, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_int8_perTensor_scale_fp32", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000001, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{2, 448}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 2043, 448}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 1}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 2043}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {1, 2}, 2, -1, true, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_int8_perTensor_scale_bf16", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000001, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{2, 448}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 2043, 448}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 1}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 2043}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {1, 2}, 2, -1, true, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_int8_perChannel_scale_int64", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000001, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{2, 448}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 2043, 448}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 2043}}, ge::DataType::DT_INT64),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 2043}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {1, 2}, 2, -1, true, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_fp8_doubleScale", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000001, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{2, 448}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("weight", {{2, 2043, 448}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{2, 1}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 2043}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {2, 1}, ge::DataType::DT_FLOAT),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {1, 2}, 2, -1, true, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_hif8_doubleScale_groupListType_1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000000, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{2, 448}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("weight", {{3, 448, 2043}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{3, 1}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 2043}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {3, 1}, ge::DataType::DT_FLOAT),
              GenTensor("group_list", {3}, ge::DataType::DT_INT64), {0, 1, 1}, 2, -1, false, false, 0, 1, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_int8_perChannel_scale_bf16_bias_int32", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000101, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{2, 448}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 2043, 448}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 2043}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 2043}}, ge::DataType::DT_BF16)},
               GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {2}, ge::DataType::DT_INT64), {1, 2}, 2, 1, true, false, 0, 1, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_int8_perChannel_scale_bf16_bias_bf16", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000101, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{128, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 504, 5}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{4, 504}}, ge::DataType::DT_BF16),
               GenTensorList("scale", {{4, 504}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{128, 504}}, ge::DataType::DT_BF16)},
               GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 0, 127, 0}, 3, 1, true, false, 0, 1, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_int8_perChannel_scale_fp32_bias_fp32", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000100, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 7168, 2048}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{4, 2048}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 2048}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4096, 2048}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 4096}, 3, 1, false, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_hifp8_perChannel_scale_fp32_splitM", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000101, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("weight", {{4, 2048, 7168}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 2048}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4096, 2048}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 4096}, 3, 1, true, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_hifp8_perChannel_scale_fp32_splitK", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000110, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{255, 4096}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("weight", {{255, 2048}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 2048}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4, 4096, 2048}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 256}, 3, 1, false, true, 2, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_fp8_perChannel_scale_fp32_splitM", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000101, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
               GenTensorList("weight", {{4, 2048, 7168}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 2048}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4096, 2048}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 4096}, 3, 1, true, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_fp8_perChannel_scale_fp32_splitK", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000110, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{255, 5}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("weight", {{255, 255}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 255}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4, 5, 255}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 256}, 3, 1, false, true, 2, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_hifp8_petensor_scale_fp32_splitK", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000010, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{255, 5}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("weight", {{255, 255}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4, 5, 255}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 256}, 3, 1, false, true, 2, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_fp8_double_scale_fp32_splitK", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000010, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{255, 5}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("weight", {{255, 255}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4, 5, 255}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {4}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 256}, 3, 1, false, true, 2, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_int8_perToken_scale_bf16_bias_fp32", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000101, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{2, 448}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 2043, 448}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{2, 2043}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{2, 2043}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 2043}}, ge::DataType::DT_BF16)},
               GenTensor("per_token_scale", {2, 448}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {2}, ge::DataType::DT_INT64), {1, 2}, 3, 1, true, false, 0, 1, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_int8_perToken_scale_bf16_bias_bf16", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000101, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{128, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 504, 5}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{4, 504}}, ge::DataType::DT_BF16),
               GenTensorList("scale", {{4, 504}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{128, 504}}, ge::DataType::DT_BF16)},
               GenTensor("per_token_scale", {4, 128}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 0, 127, 0}, 3, 1, true, false, 0, 1, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_hifp8_perToken_scale_fp32_splitM", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000101, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("weight", {{4, 2048, 7168}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 2048}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4096, 2048}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {4, 4096}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 4096}, 3, 1, true, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_hifp8_perToken_scale_fp32_splitK", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000110, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{255, 4096}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("weight", {{255, 2048}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 2048}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4, 4096, 2048}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {4, 4096}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 256}, 3, 1, false, true, 2, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_fp8_perToken_scale_fp32_splitM", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000101, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
               GenTensorList("weight", {{4, 2048, 7168}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 2048}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4096, 2048}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {4, 4096}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 4096}, 3, 1, true, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_fp8_perToken_scale_fp32_splitK", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000110, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{255, 5}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("weight", {{255, 255}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 255}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4, 5, 255}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {4, 5}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 250}, 3, 1, false, true, 2, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_fp8_GB_splitM", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000201, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
               GenTensorList("weight", {{4, 2048, 7168}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 16, 56}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4096, 2048}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {4096, 56}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 4096}, 3, 1, true, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_fp8_GB_splitM_w_false", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000200, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
               GenTensorList("weight", {{4, 7168, 2048}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 56, 16}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4096, 2048}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {4096, 56}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 4096}, 3, 1, false, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_hifp8_GB_splitM", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000201, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("weight", {{4, 2048, 7168}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 16, 56}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4096, 2048}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {4096, 56}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 4096}, 3, 1, true, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_fp8_GB_splitK", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000210, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{255, 5}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("weight", {{255, 255}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{5, 2}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4, 5, 255}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {5, 5}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 250}, 3, 1, false, true, 2, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_fp8_GB_splitK", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000210, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{255, 5}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("weight", {{255, 255}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{5, 2}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4, 5, 255}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {5, 5}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 250}, 3, 1, false, true, 2, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedNoQuantMatmul_m_sss_case_1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 10000900009000090000UL, 1)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{1, 1}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{1, 1, 1}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{1, 1}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("group_list", {1}, ge::DataType::DT_INT64), {1}, 2, -1, false, false, 0, 1, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedNoQuantMatmul_k_sss_case_1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 10000900009000090001UL, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{448, 8}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{448, 2048}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 8, 2048}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {271, 448}, 2, -1, false, true, 2, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedNoQuantMatmul_m_sss_case_2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 10000900009000090002UL, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{144, 448}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{8, 4096, 448}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{144, 4096}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("group_list", {8}, ge::DataType::DT_INT64), {18, 18, 18, 18, 18, 18, 18, 18}, 3, -1, true, false, 0, 1, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedNoQuantMatmul_m_sms_case_1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 10000900009000090000UL, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{40, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{1024, 92}, {1024, 92}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{40, 92}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {36, 4}, 3, -1, false, false, 0, 1, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedNoQuantMatmul_m_mmm_case_1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 10000900009000090000UL, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{2, 4096}, {12, 4096}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4096, 1792}, {4096, 1792}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 1792}, {12, 1792}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {2, 14}, 0, -1, false, false, -1, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_int8_perTensor_scale_fp32_bias_fp32", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000100, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 7168, 2048}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{4, 2048}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{4, 1}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4096, 2048}}, ge::DataType::DT_FLOAT16)},
               GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
               GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 5, 128, 4096}, 3, 1, false, false, 0, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedNoQuantMatmul_k_sss_case_2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 10000900009000090001UL, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{448, 2048}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{448, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 2048, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {271, 448}, 2, -1, false, true, 2, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedNoQuantMatmul_k_sss_case_3", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 10000900009000090001UL, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{448, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{448, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{1, 512, 1024}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("group_list", {1}, ge::DataType::DT_INT64), {448}, 2, -1, false, true, 2, 0, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedNoQuantMatmul_m_sss__weight_nz_case_1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 10000900009000090002UL, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{114, 448}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{2, 28, 128, 16, 16}}, ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{144, 2048}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {114, 0}, 3, -1, true, false, 0, 1, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedNoQuantMatmul_m_sms__weight_nz_case_1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 10000900009000090002UL, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{114, 448}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{28, 128, 16, 16},{28, 128, 16, 16}}, ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{144, 2048}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {57, 57}, 3, -1, true, false, 0, 1, 0),
              /* GroupListData, SplitItem, Dtype, TransposeWeight, TransposeX, GroupType, GroupListType, ActType*/
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp4_split_m_false_true", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000001, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{2, 448}}, ge::DataType::DT_FLOAT4_E1M2),
               GenTensorList("weight", {{2, 2043, 448}}, ge::DataType::DT_FLOAT4_E1M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{2, 2043, 7, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 2043}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {2, 7, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {1, 2}, 2, -1, true, false, 0, 0, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp4_split_m_false_false", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 20000000000, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{7, 810}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("weight", {{1, 810, 1782}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("bias", {{1, 1782}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{1, 13, 1782, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{7, 1782}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {7, 13, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {1}, ge::DataType::DT_INT64), {7}, 2, -1, false, false, 0, 0, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp4_group_type_error", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 20000000010, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{960, 2068}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("weight", {{960, 1014}}, ge::DataType::DT_FLOAT4_E1M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{19, 1014, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{1, 2068, 1014}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {19, 2068, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {4}, ge::DataType::DT_INT64), {0,}, 2, -1, false, true, 2, 1, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp4_split_m_transpose_error", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 20000000011, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{448, 2}}, ge::DataType::DT_FLOAT4_E1M2),
               GenTensorList("weight", {{2, 2043, 448}}, ge::DataType::DT_FLOAT4_E1M2),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{2, 2043, 7, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{2, 2043}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {2, 7, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {2}, ge::DataType::DT_INT64), {1, 2}, 2, -1, true, true, 0, 0, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp4_split_m_x_pertoken_transpose_inconsistent_error", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 20000000000, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{7, 810}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("weight", {{1, 810, 1782}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("bias", {{1, 1782}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{1, 13, 1782, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{7, 1782}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {13, 7, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {1}, ge::DataType::DT_INT64), {7}, 2, -1, false, false, 0, 0, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp4_split_m_weight_scale_transpose_insonsistent_error", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 20000000000, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{7, 810}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("weight", {{1, 810, 1782}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("bias", {{1, 1782}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{1, 1782, 13, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{7, 1782}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {7, 13, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {1}, ge::DataType::DT_INT64), {7}, 2, -1, false, false, 0, 0, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp4_split_m_bias_shape_error", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 20000000000, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{7, 810}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("weight", {{1, 810, 1782}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("bias", {{1, 2560}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{1, 1782, 13, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{7, 1782}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {7, 13, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {1}, ge::DataType::DT_INT64), {7}, 2, -1, false, false, 0, 0, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp4_split_m_scale_shape_error", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 20000000000, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{7, 810}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("weight", {{1, 810, 1782}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("bias", {{1, 1782}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{1, 15, 1782, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{7, 1782}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {7, 13, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {1}, ge::DataType::DT_INT64), {7}, 2, -1, false, false, 0, 0, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp4_split_m_pertoken_scale_shape_error", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 20000000000, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{7, 810}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("weight", {{1, 810, 1782}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("bias", {{1, 1782}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{1, 13, 1782, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{7, 1782}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {7, 64, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {1}, ge::DataType::DT_INT64), {7}, 2, -1, false, false, 0, 0, 0),
        0),
    GroupedMatmulCase(
        "GroupedQuantMM_mxfp4_split_bias_dtype_error", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, 20000000000, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{7, 810}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("weight", {{1, 810, 1782}}, ge::DataType::DT_FLOAT4_E2M1),
               GenTensorList("bias", {{1, 1782}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 13, 1782, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{7, 1782}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {7, 13, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {1}, ge::DataType::DT_INT64), {7}, 2, -1, false, false, 0, 0, 0),
        0));

INSTANTIATE_TEST_SUITE_P(GroupedMatmul_David, Ts_GroupedMatmul_WithParam_Ascend910_9591, Tc_Gmm_Tiling_Case_David);

} // namespace