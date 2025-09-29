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
 * \file test_moeTokenUnpermute_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTokenUnpermute : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenUnpermute Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenUnpermute Proto Test TearDown" << std::endl;
  }
};

TEST_F(MoeTokenUnpermute, test_infershape_bf16) {
    std::vector<int64_t> restore_shape({});
    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermute",
    { // input info
        {{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{6144, 8}, {6144, 8}}, ge::DT_BF16, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}, 
    }, 
    { // attr
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}

TEST_F(MoeTokenUnpermute, test_infershape_fp16) {
    std::vector<int64_t> restore_shape({});
    gert::InfershapeContextPara infershapeContextPara("MoeTokenUnpermute",
    { // input info
        {{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
    }, 
    { // attr
        {"padded_mode",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"restore_shape",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(restore_shape)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}
