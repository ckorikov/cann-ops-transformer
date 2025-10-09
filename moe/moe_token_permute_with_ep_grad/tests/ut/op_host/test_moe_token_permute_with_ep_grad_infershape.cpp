/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTokenPermuteWithEpGradInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithEpGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithEpGrad Proto Test TearDown" << std::endl;
    }
};
TEST_F(MoeTokenPermuteWithEpGradInferShape, MoeTokenPermuteWithEpGrad_infershape_case_0)
{
    std::vector<int64_t> range({0, 49152});
    gert::StorageShape permuted_tokens_output_d_shape = {{range[1] - range[0], 5120}, {range[1] - range[0], 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    gert::StorageShape permuted_probs_output_d_shape = {{range[1] - range[0]}, {range[1] - range[0]}};
    // output
    gert::StorageShape input_tokens_grad_shape = {{6144, 5120}, {6144, 5120}};
    gert::StorageShape input_probs_grad_shape = {{6144, 8}, {6144, 8}};
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{permuted_tokens_output_d_shape, ge::DT_BF16, ge::FORMAT_ND},
         {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
         {permuted_probs_output_d_shape, ge::DT_BF16, ge::FORMAT_ND}},
        {{input_tokens_grad_shape, ge::DT_BF16, ge::FORMAT_ND},
         {input_probs_grad_shape, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{6144, 5120}, {6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}