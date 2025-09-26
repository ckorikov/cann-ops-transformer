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
#include "../../../op_host/moe_finalize_routing_v2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeFinalizeRoutingTilingV2 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingTilingV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingTilingV2 TearDown" << std::endl;
    }
};

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRouting_tiling_float) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 30000;
    string expectTilingData = "16 16 0 16 16 1 1 1 1 1 1 1 1 1 16 16 1 1 1 16 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}