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

class MoeFinalizeRoutingV2GradProto : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2GradProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2GradProto TearDown" << std::endl;
    }
};

TEST_F(MoeFinalizeRoutingV2GradProto, shape_infer)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2Grad",
    {
        {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        // {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{15, 8}, {5, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
    // ge::op::MoeFinalizeRoutingV2Grad op;
    // std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};

    // auto grad_y_tensor_desc =
    //     create_desc_shape_range({5, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {5, 8}, ge::FORMAT_ND, shape_range);
    // op.UpdateInputDesc("grad_y", grad_y_tensor_desc);

    // auto expanded_row_idx_tensor_desc =
    //     create_desc_shape_range({15}, ge::DT_INT32, ge::FORMAT_ND, {15}, ge::FORMAT_ND, shape_range);
    // op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);

    // auto expanded_x_tensor_desc =
    //     create_desc_shape_range({15, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {15, 8}, ge::FORMAT_ND, shape_range);
    // op.UpdateInputDesc("expanded_x", expanded_x_tensor_desc);

    // auto scales_tensor_desc =
    //     create_desc_shape_range({5, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {5, 3}, ge::FORMAT_ND, shape_range);
    // op.UpdateInputDesc("scales", scales_tensor_desc);

    // auto expert_idx_tensor_desc =
    //     create_desc_shape_range({5, 3}, ge::DT_INT32, ge::FORMAT_ND, {5, 3}, ge::FORMAT_ND, shape_range);
    // op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);

    // auto bias_tensor_desc =
    //     create_desc_shape_range({8, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 8}, ge::FORMAT_ND, shape_range);
    // op.UpdateInputDesc("bias", bias_tensor_desc);

    // auto ret = InferShapeTest(op);
    // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // auto grad_expanded_x_tensor_desc = op.GetOutputDescByName("grad_expanded_x");
    // std::vector<int64_t> grad_expanded_x_tensor_expected_shape = {15, 8};
    // EXPECT_EQ(grad_expanded_x_tensor_desc.GetShape().GetDims(), grad_expanded_x_tensor_expected_shape);

    // auto grad_scales_tensor_desc = op.GetOutputDescByName("grad_scales");
    // std::vector<int64_t> grad_scales_tensor_expected_shape = {5, 3};
    // EXPECT_EQ(grad_scales_tensor_desc.GetShape().GetDims(), grad_scales_tensor_expected_shape);
}
