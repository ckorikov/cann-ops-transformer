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
 * \file ts_aclnn_grouped_matmul.cpp
 * \brief GroupedMatmul ACLNN 测试用例.
 */

#include "ts_aclnn_grouped_matmul.h"

using gmmTestParam::Ts_Aclnn_GroupedMatmul_WithParam_Ascend910_9591;
using gmmTestParam::Ts_Aclnn_GroupedMatmul_WithParam_Ascend910B3;
namespace {
TEST_P(Ts_Aclnn_GroupedMatmul_WithParam_Ascend910B3, Tc_Aclnn_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_GroupedMatmul_Aclnn_Case = ::testing::Values(
    AclnnGroupedMatmulCase("Test_GMMV4_A4W4_INT32", true, "",                         /* CaseName,Enable,DebugInfo */
              OpInfo(ControlInfo(true, true),                     /* RunTiling,RunKernel */
                     ExpectInfo(true,                             /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              AclnnGroupedMatmulParam({GenTensorList("x", {{64, 32}}, ge::DataType::DT_INT32),
                                   GenTensorList("weight", {{1, 256, 16}}, ge::DataType::DT_INT32, ge::FORMAT_FRACTAL_NZ),
                                   GenTensorList("scale", {{1,1,128}}, ge::DataType::DT_UINT64),
                                   GenTensorList("pertoken_scale", {{64,}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{64, 128}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                                   {64}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT_PERTOKEN,
                                   AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_A4W4", true, "",                         /* CaseName,Enable,DebugInfo */
              OpInfo(ControlInfo(true, true),                     /* RunTiling,RunKernel */
                     ExpectInfo(true,                             /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
              AclnnGroupedMatmulParam({GenTensorList("x", {{64, 256}}, ge::DataType::DT_INT4),
                                   GenTensorList("weight", {{1, 256, 128}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                                   GenTensorList("scale", {{1,1,128}}, ge::DataType::DT_UINT64),
                                   GenTensorList("pertoken_scale", {{64,}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{64, 128}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                                   {64}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT_PERTOKEN,
                                   AclnnGroupedMatmulVersion::V4)),

    AclnnGroupedMatmulCase("Test_GMMV4_SplitK", true, "",                         /* CaseName,Enable,DebugInfo */
                            OpInfo(ControlInfo(true, true),                     /* RunTiling,RunKernel */
                                   ExpectInfo(true,                             /* ExpectSuccess */
                                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                            AclnnGroupedMatmulParam({GenTensorList("x", {{256,64}}, ge::DataType::DT_FLOAT16),
                                                 GenTensorList("weight", {{32,64},{32,64}}, ge::DataType::DT_FLOAT16),
                                                 GenTensorList("y", {{256,64},{256,64}}, ge::DataType::DT_FLOAT16)},
                                                 GenTensor("grouped_list", {0}, ge::DataType::DT_INT64),
                                                 {}, 0, -1, false, true, 2, 1, 0, FunctionType::NO_QUANT,
                                                 AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV5_offset", true, "",       /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{64,256}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{1,256,64}}, ge::DataType::DT_INT4),
                                          GenTensorList("bias", {{1,64}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("scale", {{1,1,64}}, ge::DataType::DT_UINT64),
                                          GenTensorList("offset", {{1,1,64}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("pertoken_scale", {{64,1}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{64,64}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                                         {64}, 3, -1, false, false, 0, 1, 0, {84}, FunctionType::QUANT_PERTOKEN,
                                         AclnnGroupedMatmulVersion::V5)),
    AclnnGroupedMatmulCase("Test_GMMV4_A8W4", true, "",                         /* CaseName,Enable,DebugInfo */
                            OpInfo(ControlInfo(true, true),                     /* RunTiling,RunKernel */
                                   ExpectInfo(true,                             /* ExpectSuccess */
                                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                            AclnnGroupedMatmulParam({GenTensorList("x", {{64,256}}, ge::DataType::DT_INT8),
                                                 GenTensorList("weight", {{1,256,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                                                 GenTensorList("bias", {{1,64}}, ge::DataType::DT_FLOAT),
                                                 GenTensorList("scale", {{4,1,64}}, ge::DataType::DT_UINT64),
                                                 GenTensorList("offset", {{2}}, ge::DataType::DT_FLOAT),
                                                 GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                                                 GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),
                                                 GenTensorList("pertoken_scale", {{64,1}}, ge::DataType::DT_FLOAT),
                                                 GenTensorList("y", {{64,64}}, ge::DataType::DT_BF16)},
                                                 GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                                          {64}, 2, -1, false, false, 0, 1, 0, FunctionType::QUANT_PERTOKEN,
                                          AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_A16W4_INT32_PACKED_WEIGHT", true, "",       /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 512}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 512, 320}}, ge::DataType::DT_INT32),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_scale", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_offset", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::ANTIQUANT,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV1_SPLIT_0", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {65, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {0}, ge::DataType::DT_INT64),
                                         {}, 0, -1, false, false, -1, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V1)),
    AclnnGroupedMatmulCase("Test_GMMV1_SPLIT_3", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {15, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 2, -1, false, false, 0, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_01", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V3)),
    AclnnGroupedMatmulCase("Test_GMMV4_01", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_INT64),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_INT8)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_02", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, true, false, 0, 1, 0, FunctionType::QUANT,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_03", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_BF16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_BF16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_04", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_BF16),
                                          GenTensorList("pertoken_scale", {{84}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_BF16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT_PERTOKEN,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_05", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("pertoken_scale", {{84}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT_PERTOKEN,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_06", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{4, 2560, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_scale", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_offset", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::ANTIQUANT,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV2_07", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {15, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {15, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {0}, ge::DataType::DT_INT64),
                                         {}, 0, -1, false, false, -1, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV1_08", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {15, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 3, -1, false, false, 0, 0, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V1)),
    AclnnGroupedMatmulCase("Test_GMMV4_09", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{65, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2560, 2560}, {2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 3, -1, false, false, 0, 0, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_10", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{65, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2560, 2560}, {2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {15, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 0, -1, false, false, 0, 0, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V1)),
    AclnnGroupedMatmulCase("Test_GMMV4_11", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{1280, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{2, 1280, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {1280, 1280}, 3, -1, false, true, 2, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_12", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{1280, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{2, 1280, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {1280, 2560}, 3, -1, false, true, 2, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV4_13", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{16, 16}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 16, 10}}, ge::DataType::DT_INT4),
                                          GenTensorList("bias", {{2, 10}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_scale", {{2, 2, 10}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_offset", {{2, 2, 10}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{16, 10}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {8, 8}, 3, -1, false, false, 0, 1, 0, FunctionType::ANTIQUANT,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV1_SPLIT_0", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {65, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {0}, ge::DataType::DT_INT64),
                                         {}, 0, -1, false, false, -1, 1, 6, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV1__Error0", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{65, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 1, -1, false, false, -1, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V1)),
    AclnnGroupedMatmulCase("Test_GMMV1_Error_1", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {65, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {0}, ge::DataType::DT_INT64),
                                         {}, 0, -1, false, false, -1, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V1)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_0", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 512}, {15, 512}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{512, 256}, {512, 256}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{256}, {256}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 256}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 2, -1, false, false, 1, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_1", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{2, 128, 256}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{256, 256}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{2, 128, 256}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {128, 256}, 3, -1, false, true, 2, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_2", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{1280, 1280}, {1280, 1280}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{2, 1280, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {1280, 2560}, 3, -1, false, true, 2, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_3", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{65, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {40, 50, 65}, 2, -1, false, false, 0, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_4", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {15, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}, {5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{40, 2560}, {25, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
                                         {}, 0, -1, false, false, -1, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_5", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{50, 5120}, {15, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{50, 2560}, {15, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 65}, 2, -1, false, false, 0, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_6", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{20, 512}, {20, 512}, {25, 512}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{512, 2560}, {512, 2560}, {512, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{2560}, {2560}, {2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{65, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {20, 50, 65}, 2, -1, false, false, 0, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V2)),
    AclnnGroupedMatmulCase("Test_GMMV4_Error_0", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{2560, 1280}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 2560, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{1280, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {1280, 1280}, 3, -1, false, false, 2, 1, 0, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_1", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 512}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{512, 2560}, {512, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{2, 2560}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("pertoken_scale", {{84}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT_PERTOKEN,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV2_Error_2", true, "",                         /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_BF16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, FunctionType::QUANT,
                                         AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV5_01", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, {84,5120}, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V5)),
    AclnnGroupedMatmulCase("Test_GMMV5_2", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, {84}, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V5)),
    AclnnGroupedMatmulCase("Test_GMMV5_3", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, {-1}, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::V5)),
    AclnnGroupedMatmulCase("Test_GMMV5_4", true, "",       /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 512}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 512, 320}}, ge::DataType::DT_INT32),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_scale", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_offset", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 84}, 3, -1, false, false, 0, 1, 0, {84}, FunctionType::ANTIQUANT,
                                         AclnnGroupedMatmulVersion::V5)),
    AclnnGroupedMatmulCase("Test_GMMV_weightnz_1", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_INT32)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, {84}, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::WeightNz)),
    AclnnGroupedMatmulCase("Test_GMMV_weightnz_2", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, {84}, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::WeightNz)),
    AclnnGroupedMatmulCase("Test_GMMV_weightnz_3", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, {84}, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::WeightNz)),
    AclnnGroupedMatmulCase("Test_GMMV_weightnz_4", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 512}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("weight", {{2, 512, 320}}, ge::DataType::DT_INT32, ge::FORMAT_FRACTAL_NZ),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_scale", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_offset", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 84}, 3, -1, false, false, 0, 1, 0, {84}, FunctionType::ANTIQUANT,
                                         AclnnGroupedMatmulVersion::WeightNz)),
    AclnnGroupedMatmulCase("Test_GMMV_weightnz_5", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 512}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{2, 512, 320}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("antiquant_scale", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_offset", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_INT8)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 84}, 3, -1, false, false, 0, 1, 0, {84}, FunctionType::ANTIQUANT,
                                         AclnnGroupedMatmulVersion::WeightNz)),
    AclnnGroupedMatmulCase("Test_GMMV_weightnz_6", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 512}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{2, 512, 320}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
                                          GenTensorList("bias", {{2, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("antiquant_scale", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("antiquant_offset", {{2, 2560}}, ge::DataType::DT_FLOAT16),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_INT8)},
                                         GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                                         {50, 84}, 3, -1, false, false, 0, 1, 0, {84}, FunctionType::ANTIQUANT,
                                         AclnnGroupedMatmulVersion::WeightNz)),
    AclnnGroupedMatmulCase("Test_GMMV_weightnz_7", true, "",                              /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(false,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_BF16),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_BF16, ge::FORMAT_FRACTAL_NZ),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT)},
                                         GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                                         {50, 65, 69, 84}, 3, -1, false, false, 0, 1, 0, {84}, FunctionType::NO_QUANT,
                                         AclnnGroupedMatmulVersion::WeightNz)),
    AclnnGroupedMatmulCase("Test_GMMV_weightnz_8", true, "",                         /* CaseName,Enable,DebugInfo */
                            OpInfo(ControlInfo(true, true),                     /* RunTiling,RunKernel */
                                   ExpectInfo(false,                             /* ExpectSuccess */
                                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                            AclnnGroupedMatmulParam({GenTensorList("x", {{64,256}}, ge::DataType::DT_INT8),
                                                 GenTensorList("weight", {{1,256,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                                                 GenTensorList("bias", {{1,64}}, ge::DataType::DT_FLOAT),
                                                 GenTensorList("scale", {{4,1,64}}, ge::DataType::DT_UINT64),
                                                 GenTensorList("offset", {{2}}, ge::DataType::DT_FLOAT),
                                                 GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                                                 GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),
                                                 GenTensorList("pertoken_scale", {{64,1}}, ge::DataType::DT_FLOAT),
                                                 GenTensorList("y", {{64,64}}, ge::DataType::DT_INT8)},
                                                 GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                                          {64}, 2, -1, false, false, 0, 1, 0, FunctionType::QUANT_PERTOKEN,
                                          AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase("Test_GMMV4_grouplistTyp_2", true, "",                  /* CaseName,Enable,DebugInfo */
                           OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                                  ExpectInfo(true,                                 /* ExpectSuccess */
                                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                           AclnnGroupedMatmulParam({GenTensorList("x", {{84, 5120}}, ge::DataType::DT_INT8),
                                          GenTensorList("weight", {{4, 5120, 2560}}, ge::DataType::DT_INT8),
                                          GenTensorList("bias", {{4, 2560}}, ge::DataType::DT_INT32),
                                          GenTensorList("scale", {{4, 2560}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("pertoken_scale", {{84}}, ge::DataType::DT_FLOAT),
                                          GenTensorList("y", {{84, 2560}}, ge::DataType::DT_FLOAT16)},
                                          GenTensor("grouped_list", {4, 2}, ge::DataType::DT_INT64),
                                          {0, 50, 1, 10, 2, 4, 3, 5}, 3, -1, false, false, 0, 2, 0,
                                          FunctionType::QUANT_PERTOKEN, AclnnGroupedMatmulVersion::V4))

);

INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_Aclnn_GroupedMatmul_WithParam_Ascend910B3, Tc_GroupedMatmul_Aclnn_Case);

TEST_P(Ts_Aclnn_GroupedMatmul_WithParam_Ascend910_9591, Tc_Aclnn_GroupedMatmul_9591)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_Gmm_Aclnn_David_Case = ::testing::Values(
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_0", true, "",                          /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{576, 512}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("weight", {{4, 512, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("scale", {{4, 16, 7168}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{576, 16}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{576, 7168}}, ge::DataType::DT_BF16)},
                                GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {8, 181, 475, 576}, 3, -1, true,
                                false, 0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_1", true, "",                          /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{16, 512}}, ge::DataType::DT_FLOAT8_E5M2),
                                 GenTensorList("weight", {{1, 512, 7168}}, ge::DataType::DT_FLOAT8_E5M2),
                                 GenTensorList("scale", {{1, 16, 7168}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{16, 16}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{16, 7168}}, ge::DataType::DT_BF16)},
                                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {16}, 3, -1, true, false, 0, 1,
                                0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_2", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                         ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                         ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{256, 0}}, ge::DataType::DT_FLOAT8_E5M2),
                                GenTensorList("weight", {{2, 0, 7168}}, ge::DataType::DT_FLOAT8_E5M2),
                                GenTensorList("scale", {{2, 0, 7168}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{256, 0}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{256, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {128, 256}, 3, -1, true, false,
                               0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_3", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                         ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                         ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{64, 512}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                GenTensorList("weight", {{64, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                GenTensorList("scale", {{1, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{1, 512}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{1, 512, 4096}}, ge::DataType::DT_FLOAT)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {64}, 3, -1, false, true,
                                2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_4", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{256, 256}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                GenTensorList("weight", {{256, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                GenTensorList("scale", {{1, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{1, 256}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{1, 256, 4096}}, ge::DataType::DT_FLOAT)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {256}, 3, -1, false, true,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_5", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{512, 2048}}, ge::DataType::DT_INT8),
                                GenTensorList("weight", {{5, 2048, 7168}}, ge::DataType::DT_INT8),
                                GenTensorList("scale", {{5, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("pertoken_scale", {{512}}, ge::DataType::DT_FLOAT),
                                GenTensorList("y", {{512, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {5}, ge::DataType::DT_INT64), {128, 128, 128, 64, 64}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::QUANT_PERTOKEN, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_6", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{1024, 2048}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                GenTensorList("weight", {{4, 2048, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                GenTensorList("scale", {{4, 7168}}, ge::DataType::DT_UINT64),
                                GenTensorList("y", {{1024, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {256, 512, 768, 1024}, 3, -1, true, false,
                                   0, 0, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_7", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{512, 7168}}, ge::DataType::DT_HIFLOAT8),
                                GenTensorList("weight", {{3, 7168, 4096}}, ge::DataType::DT_HIFLOAT8),
                                GenTensorList("scale", {{3, 1}}, ge::DataType::DT_FLOAT),
                                GenTensorList("pertoken_scale", {{3, 1}}, ge::DataType::DT_FLOAT),
                                GenTensorList("y", {{512, 4096}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {64, 256, 512}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::QUANT_PERTOKEN, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_8", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{7168, 144}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                GenTensorList("weight", {{144, 2048}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                GenTensorList("scale", {{1, 2048}}, ge::DataType::DT_FLOAT),
                                GenTensorList("pertoken_scale", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("y", {{7168, 2048}}, ge::DataType::DT_FLOAT16)},
                               GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {144}, 3, -1, false, true,
                                   2, 0, 0, FunctionType::QUANT_PERTOKEN, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_9", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 256}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                GenTensorList("weight", {{256, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                GenTensorList("scale", {{2, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {128, 256}, 3, -1, false, true,
                                   2, 0, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_10", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_HIFLOAT8),
                                GenTensorList("weight", {{128, 7168}}, ge::DataType::DT_HIFLOAT8),
                                GenTensorList("scale", {{2, 1}}, ge::DataType::DT_FLOAT),
                                GenTensorList("pertoken_scale", {{2, 1}}, ge::DataType::DT_FLOAT),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {64, 128}, 3, -1, false, true,
                                   2, 0, 0, FunctionType::QUANT_PERTOKEN, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_11", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_12", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 7168, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 7168, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, true, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_13", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 7168, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("scale", {{1, 7168, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, true, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_14", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 7168, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("scale", {{1, 7168, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_group_type_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   2, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_transpose_false_true_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 4096}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, true,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_scale_dtype_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_bias_dtype_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_weight_dtype_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_per_token_scale_dtype_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_output_dtype_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_INT8)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_scale_transpose_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 7168, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_split_m_weight_dim_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_scale_dim_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_pertoken_dim_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_k_not_equal_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 256, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_bias_dim_value_error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7169}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_scale_dim_value__error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 1, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4096, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp4_per_token_dim_value__error", true, "",                          /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 128}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("weight", {{1, 128, 7168}}, ge::DataType::DT_FLOAT4_E1M2),
                                GenTensorList("bias", {{1, 7168}}, ge::DataType::DT_FLOAT),
                                GenTensorList("scale", {{1, 2, 7168, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{4095, 2, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{4096, 7168}}, ge::DataType::DT_BF16)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_x_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 7168}}, ge::DataType::DT_INT8),
                                GenTensorList("weight", {{1, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                GenTensorList("scale", {{1, 224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("pertoken_scale", {{128, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                GenTensorList("y", {{128, 4096}}, ge::DataType::DT_FLOAT)},
                               GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_weight_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{1, 4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{1, 4096, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{128, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{128, 4096}}, ge::DataType::DT_FLOAT)},
                                   GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_scale_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{1, 4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{1, 4096, 224}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{128, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{128, 4096}}, ge::DataType::DT_FLOAT)},
                                   GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_per_token_scale_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{1, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{1, 224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{128, 224}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{128, 4096}}, ge::DataType::DT_FLOAT)},
                                   GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_output_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{1, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{1, 224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{128, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{128, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_N0_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7168, 0}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 224, 0}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 0}}, ge::DataType::DT_BF16)},
                                   GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_group_num_inconsistent_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{3, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{2, 224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{128, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{128, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_scale_n_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 224, 1024}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_pertoken_m_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{1024, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_scale_k_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 200, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_pertoken_k_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096, 200}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_k_not_equal_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7167, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_scale_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_pertoken_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxpf8_split_m_weight_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_split_k_weight_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, true,
                                   0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_transpose_true_true_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, true, true,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxfp8_transpose_false_false_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 224, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_Error_NO_QUANT_0", true, "",                         /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                         /* RunTiling,RunKernel */
              ExpectInfo(false,                                /* ExpectSuccess */
                         ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                         ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{256, 1280}}, ge::DataType::DT_FLOAT16),
                      GenTensorList("weight", {{2, 256, 256}}, ge::DataType::DT_FLOAT16),
                      GenTensorList("y", {{1280, 256}}, ge::DataType::DT_FLOAT16)},
                     GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
                     {1280, 1280}, 3, -1, false, false, 2, 1, 0, FunctionType::NO_QUANT,
                     AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_fp8_scale_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_bf16_scale_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_BF16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_fp16_scale_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_BF16),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_hifloat8_pertoken_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096, 224}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_scale_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_UINT64),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_bias_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{32, 4096}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_bias_dtype_bf16_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{32, 4096}}, ge::DataType::DT_BF16),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_bias_dtype_fp16_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{32, 4096}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_BF16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_output_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_hifloat8_output_dtype_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_HIFLOAT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_scale_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{32, 224, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_pertoken_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{1, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{1, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{1, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT_PERTOKEN, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_bias_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{32, 1, 4096}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_scale_shape_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{32, 409}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_pertoken_shape_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{1, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{1, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{1}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {4096}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT_PERTOKEN, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_bias_shape_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{32, 1}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_typeK_scale_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{7168, 4096}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, true,
                                   2, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_typeK_pertoken_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{7168, 4096}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, true,
                                   2, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_typeM_transpose_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, true,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_typeK_pertoken_first_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{7168, 4096}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{31, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{32, 4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, true,
                                   2, 1, 0, FunctionType::QUANT_PERTOKEN, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_typeK_pertoken_second_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{7168, 4096}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{32, 4095}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{32, 4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, true,
                                   2, 1, 0, FunctionType::QUANT_PERTOKEN, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_typeK_transpose_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{7168, 4096}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, true, false,
                                   2, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_doublescale_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{32, 1}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{32, 1}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_hifloat8_bias_not_null_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{32, 7168, 4096}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("bias", {{32, 4096}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("scale", {{32, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_activation_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 1, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_multi_weight_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("weight", {{128, 256}, {128, 256}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_a16w4_not_support_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_INT4),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
     AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_a16fp8_with_offset_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_FLOAT8_E5M2),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV3_9591_antiquant_a16fp8_not _support_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_FLOAT8_E5M2),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V3)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_a16w8_nz_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_FLOAT16),
                                GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_weight_dim_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("weight", {{128, 256}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_x_transpose_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, false, true,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_weight_not_transpose_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_N_exceeds_limit_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("weight", {{2, 128, 65536}}, ge::DataType::DT_INT8),
                                   GenTensorList("antiquant_scale", {{2, 65536}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_offset", {{2, 65536}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 65536}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_K_32_not_align_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 120}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("weight", {{2, 120, 256}}, ge::DataType::DT_INT8),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_bias_invalid_type_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_BF16),
                                   GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_BF16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_BF16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_BF16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_offset_invalid_type_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_BF16),
                                   GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_BF16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_BF16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_BF16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_scale_invalid_type_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_BF16),
                                   GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_BF16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_FLOAT16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_BF16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_BF16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_offset_invalid_shape_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_BF16),
                                   GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_BF16),
                                   GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_BF16),
                                   GenTensorList("antiquant_offset", {{2, 128}}, ge::DataType::DT_BF16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_BF16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_antiquant_scale_invalid_shape_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{32, 128}}, ge::DataType::DT_BF16),
                                   GenTensorList("weight", {{2, 128, 256}}, ge::DataType::DT_INT8),
                                   GenTensorList("bias", {{2, 256}}, ge::DataType::DT_BF16),
                                   GenTensorList("antiquant_scale", {{1, 256}}, ge::DataType::DT_BF16),
                                   GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_BF16),
                                   GenTensorList("y", {{32, 256}}, ge::DataType::DT_BF16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {16, 16}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::ANTIQUANT, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_pertoken", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{127, 3}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{127, 128}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{2, 1}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{2, 3}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{2, 3, 128}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {64, 127}, 2, -1, false, true,
                                   2, 1, 0, FunctionType::QUANT_PERTOKEN, AclnnGroupedMatmulVersion::V4)),

    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_pertile_scale_k_not_correct_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 256}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{2, 256, 384}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{2, 1, 3}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{128, 2}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{128, 384}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {64, 128}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_pertile_scale_n_not_correct_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 256}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{2, 256, 384}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{2, 2, 2}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{128, 2}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{128, 384}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {64, 128}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_pertile_perTokenscale_m_not_correct_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 256}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{2, 256, 384}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{2, 2, 3}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{1, 2}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{128, 384}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {64, 128}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_pertile_perTokenscale_k_not_correct_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 256}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{2, 256, 384}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{2, 2, 3}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{128, 1}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{128, 384}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {64, 128}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_pertile_scale_dim_not_correct_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 256}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{2, 256, 384}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{2, 3}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{128, 1}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{128, 384}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {64, 128}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_pertile_k_group_scale_dim_not_correct_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{256, 128}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{256, 384}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{3}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4, 128}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{2, 128, 384}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {128, 256}, 3, -1, true, false,
                                   2, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_pertile_k_group_perTokenscale_shape_not_correct_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 256, }}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{256, 384}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{4, 3}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{128,3, }}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{2, 128, 384}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {128, 256}, 3, -1, false, true,
                                   2, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_pertile_perTokenscale_dim_not_correct_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 256, }}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{256, 384}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{4, 3}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{128}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{2, 128, 384}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {128, 256}, 3, -1, false, true,
                                   2, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_pertile_perTokenscale_dtype_not_correct_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128,256, }}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{256, 384}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{4, 3}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{128,4, }}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{2, 128, 384}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {128, 256}, 3, -1, false, true,
                                   2, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_pertile_scale_dtype_not_correct_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{128, 256,}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{256, 384}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{4, 3}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{128, 4, }}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{2, 128, 384}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {2}, ge::DataType::DT_INT64), {128, 256}, 3, -1, false, true,
                                   2, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_TYPEM_pertile_scale_dtype_not_correct_error", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{972, 51,}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{3, 51, 44}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{3, 1, 1}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{972, 2, }}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{972, 44}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {3}, ge::DataType::DT_INT64), {0, 0, 972}, 3, -1, false, false,
                                   0, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_mx_tpyek_scale_shape_last_dim_error", true, "",                          /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{1, 512}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("weight", {{512, 1}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("scale", {{12, 1, 3}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{12, 1 ,2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{4, 1, 1}}, ge::DataType::DT_BF16)},
                                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {10, 10, 10, 10}, 3, -1, false,
                                 true, 2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_mx_tpyek_pertoken_shape_second_dim_error", true, "",                          /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{1, 512}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("weight", {{512, 1}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("scale", {{12, 1, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{12, 2 ,2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{4, 1, 1}}, ge::DataType::DT_BF16)},
                                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {10, 10, 10, 10}, 3, -1, false,
                                 true, 2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_mx_tpyek_pertoken_shape_first_dim_error", true, "",                          /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{1, 512}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("weight", {{512, 1}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("scale", {{12, 1, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{11, 1 ,2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{4, 1, 1}}, ge::DataType::DT_BF16)},
                                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {10, 10, 10, 10}, 3, -1, false,
                                 true, 2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_mx_tpyem_scale_shape_dim_num_error", true, "",                          /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{1, 512}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("weight", {{512, 1}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("scale", {{4, 16, 1}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{1 ,16}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{1, 1}}, ge::DataType::DT_BF16)},
                                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {10, 10, 10, 10}, 3, -1, true,
                                 false, 0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_mx_tpyem_scale_shape_g_dim_error", true, "",                          /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{1, 512}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("weight", {{4, 512, 1}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("scale", {{3, 16, 1}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{1 ,16}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{1, 1}}, ge::DataType::DT_BF16)},
                                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {10, 10, 10, 10}, 3, -1, true,
                                 false, 0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_mx_tpyem_scale_shape_k_dim_error", true, "",                          /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{1, 512}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("weight", {{4, 512, 1}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("scale", {{4, 16, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{1 ,16}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{1, 1}}, ge::DataType::DT_BF16)},
                                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {10, 10, 10, 10}, 3, -1, true,
                                 false, 0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_mx_tpyem_scale_shape_n_dim_error", true, "",                          /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{1, 512}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("weight", {{4, 512, 1}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("scale", {{4, 15, 1}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{1 ,16}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{1, 1}}, ge::DataType::DT_BF16)},
                                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {10, 10, 10, 10}, 3, -1, true,
                                 false, 0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),

    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_pertoken_input_for_pertensor_quant_mode_shape_g", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{4, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{4, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_BF16)},
                                   GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {10, 10, 10, 10}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
   AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_int8_pertoken_input_for_pertensor_quant_mode_shape_g_1", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_INT8),
                                   GenTensorList("weight", {{4, 7168, 4096}}, ge::DataType::DT_INT8),
                                   GenTensorList("scale", {{4, 4096}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{4, 1}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_BF16)},
                                   GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {10, 10, 10, 10}, 3, -1, true, false,
                                   0, 1, 0, FunctionType::QUANT, AclnnGroupedMatmulVersion::V4)),
   AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_mx_tpyek_pertoken_shape_success", true, "",                          /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{1, 512}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("weight", {{512, 1}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("scale", {{12, 1, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{1, 12 ,2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{4, 1, 1}}, ge::DataType::DT_BF16)},
                                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {10, 10, 10, 10}, 3, -1, false,
                                 true, 2, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_mx_tpyem_scale_shape_success", true, "",                          /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{1, 512}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("weight", {{4, 512, 1}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("scale", {{4, 8, 1, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{1 ,8, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{1, 1}}, ge::DataType::DT_BF16)},
                                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64), {10, 10, 10, 10}, 3, -1, true,
                                 false, 0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_mxpf8_special_scale_shape", true, "",               /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
               ExpectInfo(false,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{1, 1}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("weight", {{1, 1, 1}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                 GenTensorList("scale", {{1, 1, 1, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("pertoken_scale", {{1, 1, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                 GenTensorList("y", {{1, 1}}, ge::DataType::DT_INT8)},
                                 GenTensor("grouped_list", {1}, ge::DataType::DT_INT64), {1}, 3, -1, true, false,
                                 0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
        "Test_GMMV4_9591_TYPEM_pertile_success", true, "",               /* CaseName,Enable,DebugInfo */
        OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnGroupedMatmulParam({GenTensorList("x", {{5, 76}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("weight", {{1, 76, 2019}}, ge::DataType::DT_HIFLOAT8),
                                   GenTensorList("scale", {{1, 1, 16}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("pertoken_scale", {{5, 1}}, ge::DataType::DT_FLOAT),
                                   GenTensorList("y", {{5, 2019}}, ge::DataType::DT_FLOAT16)},
                                   GenTensor("grouped_list", {3}, ge::DataType::DT_INT64), {0, 0, 3}, 3, -1, true, false,
                                   0, 0, 0, FunctionType::PERTILE, AclnnGroupedMatmulVersion::V4)),
    AclnnGroupedMatmulCase(
       "Test_GMMV4_9591_mxpf8_split_m_false_false_case", true, "",               /* CaseName,Enable,DebugInfo */
       OpInfo(ControlInfo(true, false),                        /* RunTiling,RunKernel */
              ExpectInfo(false,                                 /* ExpectSuccess */
                            ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                            ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
       AclnnGroupedMatmulParam({GenTensorList("x", {{4096, 7168}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("weight", {{1, 7168, 4096}}, ge::DataType::DT_FLOAT8_E4M3FN),
                                   GenTensorList("scale", {{1, 112, 4096, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("pertoken_scale", {{4096, 112, 2}}, ge::DataType::DT_FLOAT8_E8M0),
                                   GenTensorList("y", {{4096, 4096}}, ge::DataType::DT_INT8)},
                                   GenTensor("grouped_list", {32}, ge::DataType::DT_INT64), {64}, 3, -1, false, false,
                                   0, 1, 0, FunctionType::MXFP, AclnnGroupedMatmulVersion::V4)));
INSTANTIATE_TEST_SUITE_P(GroupedMatmulDavid, Ts_Aclnn_GroupedMatmul_WithParam_Ascend910_9591, Tc_Gmm_Aclnn_David_Case);
}  // namespace