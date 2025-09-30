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
 * \file test_moeTokenUnpermuteWithEpGrad_infershape.cpp
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTokenUnpermuteWithEpGradInferShape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithEpGrad Proto Test TearDown" << std::endl;
    }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape) {
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

TEST_F(MoeTokenUnpermuteWithEpGradInferShape, MoeTokenUnpermuteWithEpGrad_infershape_case_0)
{
    gert::StorageShape permuted_tokens_shape = {{30, 64}, {30, 64}};
    gert::StorageShape unpermuted_output_d_shape = {{10, 64}, {10, 64}};
     gert::StorageShape blank_shape = {};
    gert::StorageShape sorted_indices_shape = {
        {
            30,
        },
        {
            30,
        }};
    gert::StorageShape probs_shape = {{10, 3}, {10, 3}};
    // output
    gert::StorageShape permuted_tokens_grad_shape = {{30, 64}, {30, 64}};
    gert::StorageShape probs_grad_shape = {{10, 3}, {10, 3}};

    auto holder =
        gert::InferShapeContextFaker()
            .SetOpType("MoeTokenUnpermuteWithEpGrad")
            .NodeIoNum(4, 2)
            //.IrInstanceNum({1, 1, 1, 1})
            //.InputShapes({&unpermuted_output_d_shape, &sorted_indices_shape, &permuted_tokens_shape, &probs_shape})
            .OutputShapes({&permuted_tokens_grad_shape, &probs_grad_shape})
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputTensors({(gert::Tensor *)&unpermuted_output_d_shape})
            .InputTensors({(gert::Tensor *)&sorted_indices_shape})
            .InputTensors({(gert::Tensor *)&permuted_tokens_shape})
            .InputTensors({(gert::Tensor *)&probs_shape})
            .Attr("padded_mode", bool(false))
            .Build();

    gert::InferShapeContext* context = holder.GetContext();
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeTokenUnpermuteWithEpGrad")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedPermutedTokensGradShape = {30, 64};
    std::vector<int64_t> expectedProbsGradShape = {10, 3};
    auto permutedTokensGradShape = context->GetOutputShape(0);
    auto probsGradShape = context->GetOutputShape(1);
    EXPECT_EQ(ToVector(*permutedTokensGradShape), expectedPermutedTokensGradShape);
    EXPECT_EQ(ToVector(*probsGradShape), expectedProbsGradShape);
}