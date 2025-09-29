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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/moe_token_permute_with_ep_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeTokenPermuteWithEpTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithEpTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithEpTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_01)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    std::vector<int64_t> range({1, 5});
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithEp", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{6,}, {6,}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{4,}, {4,}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {
                                                {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
                                                {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 5; // tilngkey
    string expectTilingData = "1 2 5 16 3 1 6 1 6 6 6 1 6 6 8160 0 0 1024 2 2 0 1 0 10 2 2 2340 4680 14040 1 1 1 1 1 0 4680 2 2340 1 1 4 1 5 74880 14048 14048 ";
    std::vector<size_t> expectWorkspaces = {16777376}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_02)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    std::vector<int64_t> range({0, 4096});
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithEp", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{4096, 5}, {4096, 5}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096, 3}, {4096, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                {{{4096, 3}, {4096, 3}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{4096, 5}, {4096, 5}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096 * 3}, {4096 * 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {
                                                {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
                                                {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 6; // tilngkey
    string expectTilingData = "64 4096 5 16 3 4 3072 1 3072 3072 3072 1 3072 3072 8128 0 0 1024 64 64 0 64 64 10 2 2 2340 4680 14040 1 1 1 1 64 1 64 1 64 1 64 4096 0 4096 74880 14048 14048 ";
    std::vector<size_t> expectWorkspaces = {16977920}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteWithEpTiling, moe_token_permute_with_ep_tiling_error_h)
{
    optiling::MoeTokenPermuteWithEpCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    std::vector<int64_t> range({0, 4096});
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithEp", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{4096, 5}, {4096, 5}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096, 3}, {4096, 3}}, ge::DT_INT64, ge::FORMAT_ND},
                                                {{{4096, 3}, {4096, 3}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{4096, 5}, {4096, 5}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{4096 * 3}, {4096 * 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{4096}, {4096}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {
                                                {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
                                                {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 6; // tilngkey
    string expectTilingData = "64 4096 5 16 3 4 3072 1 3072 3072 3072 1 3072 3072 8128 0 0 1024 64 64 0 64 64 10 2 2 2340 4680 14040 1 1 1 1 64 1 64 1 64 1 64 4096 0 4096 74880 14048 14048 ";
    std::vector<size_t> expectWorkspaces = {16977920}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}