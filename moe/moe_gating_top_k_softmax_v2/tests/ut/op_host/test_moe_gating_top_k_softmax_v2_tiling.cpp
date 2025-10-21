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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/moe_gating_top_k_softmax_v2_tiling.h"

using namespace std;

class MoeGatingTopKSoftmaxV2Tiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2Tiling TearDown" << std::endl;
    }
};

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_001) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101031;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_002) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                                {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101111;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_003) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101010;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_006) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_007) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24}, {1, 24}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_008) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24}, {1, 24}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_009) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 1, 5000}, {1, 1, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_010) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                                {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101110;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_011) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                                {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101110;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}
