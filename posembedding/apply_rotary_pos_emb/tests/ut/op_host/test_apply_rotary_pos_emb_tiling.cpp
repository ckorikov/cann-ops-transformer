/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/apply_rotary_pos_emb_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class ApplyRotaryPosEmbTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyRotaryPosEmbTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyRotaryPosEmbTiling TearDown" << std::endl;
    }
};

TEST_F(ApplyRotaryPosEmbTiling, test_tiling_fp16_001)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo,
                                              "Ascend910B",
                                              40,
                                              196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "24 128 64 0 0 0 0 0 0 0 3072 3072 256 256 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 1 8 4 0 0 128 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, test_tiling_bf16_001)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo,
                                              "Ascend910B",
                                              40,
                                              196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "24 128 64 0 0 0 0 0 0 0 6144 6144 256 512 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 2 16 8 0 0 64 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, test_tiling_fp32_001)
{
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                              {
                                                  // input info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {// attr
                                               {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")}},
                                              &compileInfo,
                                              "Ascend910B",
                                              40,
                                              196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "24 128 64 0 0 0 0 0 0 0 6144 6144 512 512 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 2 16 8 0 0 64 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, test_tiling_bf16_TND) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{24, 11, 128}, {24, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{24, 1, 128},  {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{24, 1, 128},  {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{24, 1, 128},  {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{24, 11, 128}, {24, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{24, 1, 128},  {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "24 128 64 0 0 0 0 0 0 0 6144 6144 256 512 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 2 16 8 0 0 64 ";
    std::vector<size_t> expectWorkspaces = {16*1024*1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, test_tiling_bf16_TND_key1) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{24, 11, 128}, {24, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{24, 1, 128},  {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{24, 1, 128},  {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{24, 1, 128},  {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{24, 11, 128}, {24, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{24, 1, 128},  {24, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "24 128 64 0 0 0 0 0 0 0 6144 6144 256 512 0 0 0 0 0 1408 128 128 1408 128 128 12 1536 2 16 8 0 0 64 ";
    std::vector<size_t> expectWorkspaces = {16*1024*1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, test_tiling_fp16_TND_key3) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 1, 128},  {4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 1, 128},  {4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "40 128 64 4 3 3 3 1 3 3 40960 30720 1024 768 25 19 1 0 0 263680 263680 13184 2560 2560 128 40 320 1280 8 160 160 4 128 ";
    std::vector<size_t> expectWorkspaces = {16*1024*1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, test_tiling_bf16_TND_key4) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096, 1, 128},  {4096, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096, 1, 128},  {4096, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096, 16, 128}, {4096, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "40 128 64 2 1 1 2 2 1 1 32768 32768 512 1024 51 39 0 0 0 210944 210944 13184 2048 2048 128 32 512 1024 16 128 128 8 64 ";
    std::vector<size_t> expectWorkspaces = {16*1024*1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_small_wrong) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1, 8, 1, 128}, {1, 8, 1, 128}}  , ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1, 8, 1, 128}, {1, 8, 1, 128}}  , ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_small_wrong) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 8, 1, 128}, {1, 8, 1, 128}}  , ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 8, 1, 128}, {1, 8, 1, 128}}  , ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 8, 48, 128}, {1, 8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_small_wrong) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{1, 8, 96, 128}, {1, 8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 8, 96, 128}, {1, 8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 8, 1,  128}, {1, 8, 1, 128}} , ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 8, 1,  128}, {1, 8, 1, 128}} , ge::DT_FLOAT16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{1, 8, 96, 128}, {1, 8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 8, 96, 128}, {1, 8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_wrong) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4, 1024, 38, 128}, {4, 1024, 38, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4, 1024, 39, 128}, {4, 1024, 39, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4, 1024, 1,  128}, {4, 1024, 1, 128}} , ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4, 1024, 1,  128}, {4, 1024, 1, 128}} , ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4, 1024, 38, 128}, {4, 1024, 38, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4, 1024, 39, 128}, {4, 1024, 39, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_wrong) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4, 1024, 38, 128}, {4, 1024, 38, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4, 1024, 39, 128}, {4, 1024, 39, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4, 1024, 1,  128}, {4, 1024, 1, 128}} , ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4, 1024, 1,  128}, {4, 1024, 1, 128}} , ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4, 1024, 38, 128}, {4, 1024, 38, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4, 1024, 39, 128}, {4, 1024, 39, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_wrong) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4, 1024, 76, 128}, {4, 1024, 76, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4, 1024, 77, 128}, {4, 1024, 77, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4, 1024, 1,  128}, {4, 1024, 1, 128}} , ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4, 1024, 1,  128}, {4, 1024, 1, 128}} , ge::DT_FLOAT16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4, 1024, 76, 128}, {4, 1024, 76, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4, 1024, 77, 128}, {4, 1024, 77, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_small_wrong_TND) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{8, 48, 128}, {8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 48, 128}, {8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 1, 128},  {8, 1, 128}}  , ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 1, 128},  {8, 1, 128}}  , ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{8, 48, 128}, {8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 48, 128}, {8, 48, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_small_wrong_TND) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{8, 48, 128}, {8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 48, 128}, {8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 1, 128},  {8, 1, 128}}  , ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 1, 128},  {8, 1, 128}}  , ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{8, 48, 128}, {8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 48, 128}, {8, 48, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_small_wrong_TND) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{8, 96, 128}, {8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 96, 128}, {8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 1,  128}, {8, 1, 128}} , ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 1,  128}, {8, 1, 128}} , ge::DT_FLOAT16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{8, 96, 128}, {8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{8, 96, 128}, {8, 96, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_bf16_wrong_TND) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4096, 38, 128}, {4096, 38, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096, 39, 128}, {4096, 39, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}} , ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}} , ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4096, 38, 128}, {4096, 38, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096, 39, 128}, {4096, 39, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp32_wrong_TND) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4096, 38, 128}, {4096, 38, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 39, 128}, {4096, 39, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}} , ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}} , ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4096, 38, 128}, {4096, 38, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 39, 128}, {4096, 39, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_fp16_wrong_TND) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4096, 76, 128}, {4096, 76, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 77, 128}, {4096, 77, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}} , ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}} , ge::DT_FLOAT16, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4096, 76, 128}, {4096, 76, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 77, 128}, {4096, 77, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}


TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_shape_is_2) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{20, 128}, {20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{20, 128}, {20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_shape_dim_num_is_diff) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4, 1024, 20, 128}, {4, 1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4, 1024, 20, 128}, {4, 1024, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_shape_TBS_is_diff) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{2048, 20, 128}, {2048, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{2048, 20, 128}, {2048, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_shape_cos_N_isnot_1) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20,  128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_shape_D_isnot_128) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4096, 20, 64}, {4096, 20, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 64}, {4096, 20, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  64}, {4096, 1,  64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  64}, {4096, 1,  64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4096, 20, 64}, {4096, 20, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 64}, {4096, 20, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_layout_is_3) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_layout_is_diff) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_Dtype_is_int32) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}},  ge::DT_INT32, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}},  ge::DT_INT32, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbTiling, arpe_wrong_Dtype_is_diff) {
    optiling::ApplyRotaryPosEmbCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ApplyRotaryPosEmb",
                                            { // input info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 1,  128}, {4096, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // output info
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 20, 128}, {4096, 20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            }, 
                                            { // attr
                                                {"layout",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                            },
                                            &compileInfo,
                                            "Ascend910B",
                                            40,
                                            196608);
 
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
