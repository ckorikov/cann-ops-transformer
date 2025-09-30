/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeInitRoutingQuantV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeInitRoutingQuantV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeInitRoutingQuantV2 TearDown" << std::endl;
  }
};

TEST_F(MoeInitRoutingQuantV2, moe_init_routing_quant_v2_infer_shape_01)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingQuantV2",
    { // input info
        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    }, 
    { // attr
        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_tokens_count_or_cumsum_flag",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_tokens_before_capacity_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}