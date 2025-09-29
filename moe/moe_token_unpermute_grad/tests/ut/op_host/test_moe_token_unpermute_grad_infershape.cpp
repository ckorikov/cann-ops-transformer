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
 * \file test_moeTokenUnpermuteGrad_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTokenUnpermuteGradInferShape : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenUnpermuteGradInferShape SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenUnpermuteGradInferShape TearDown" << std::endl;
  }
};


TEST_F(MoeTokenUnpermuteGradInferShape, MoeTokenUnpermuteGrad_infershape_case_0)
{
    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermuteGrad",
    {
        {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}
    },
    {
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}
    },
    {
        {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{30, 64}, {10, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}