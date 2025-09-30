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
 * \file test_moe_token_unpermute_with_ep_grad_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"

#include "../../../op_host/moe_token_unpermute_with_ep_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;

class MoeTokenUnpermuteWithEpGradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpGradTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenUnpermuteWithEpGradTiling, test_tiling_prob_not_none_bf16_001)
{
    // compile info
    struct MoeTokenUnpermuteWithEpGradCompileInfo {
    };
    MoeTokenUnpermuteWithEpGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithEpGrad",
		                              {
                                              {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND},
					                  },
                                      {
                                        {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND},
                                      },
                                      {
                                        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                        {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({1,1})},
                                        {"range", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({-1,-1})},
                                        {"topk_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                      },
                                             &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "10 3 64 29 10 54 1 0 3 0 256 1 64 3 3 16 262144 29 29 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,expectTilingKey,expectTilingData,expectWorkspaces);
}