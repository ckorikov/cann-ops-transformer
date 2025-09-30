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
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_quant_v2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

struct MoeInitRoutingQuantV2CompileInfo {};

class MoeInitRoutingQuantV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeInitRoutingQuantV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeInitRoutingQuantV2Tiling TearDown" << std::endl;
  }
};

TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_01) {
  MoeInitRoutingQuantV2CompileInfo compileInfo = {};
  gert::TilingContextPara tilingContextPara("MoeInitRoutingQuantV2",
                                            {
                                              {{{8, 30}, {8, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{8, 6}, {8, 6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{48, 30}, {48, 30}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{48}, {48}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{48}, {48}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                              {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                              {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_count_or_cumsum_flag",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              {"expert_tokens_before_capacity_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
                                            },
                                            &compileInfo);

  uint64_t expectTilingKey = 10000;
  string expectTilingData = "64 8 30 6 6 8 0 0 0 0 1 0 1 48 1 48 48 48 1 48 48 6912 0 2048 48 0 1 1 1 1 1 1 0 0 0 0 0 48 0 1 1 1 1 1 1 1 1 30 30 1 48 48 1 1 1 1 1 1 1 1 30 30 1 ";
  std::vector<size_t> expectWorkspaces = {16779296};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
