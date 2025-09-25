/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "fusion_ops.h"
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

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
    ge::op::MoeFinalizeRoutingV2Grad op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};

    auto grad_y_tensor_desc =
        create_desc_shape_range({5, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {5, 8}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("grad_y", grad_y_tensor_desc);

    auto expanded_row_idx_tensor_desc =
        create_desc_shape_range({15}, ge::DT_INT32, ge::FORMAT_ND, {15}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);

    auto expanded_x_tensor_desc =
        create_desc_shape_range({15, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {15, 8}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_x_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({5, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {5, 3}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_idx_tensor_desc =
        create_desc_shape_range({5, 3}, ge::DT_INT32, ge::FORMAT_ND, {5, 3}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({8, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 8}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto ret = InferShapeTest(op);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto grad_expanded_x_tensor_desc = op.GetOutputDescByName("grad_expanded_x");
    std::vector<int64_t> grad_expanded_x_tensor_expected_shape = {15, 8};
    EXPECT_EQ(grad_expanded_x_tensor_desc.GetShape().GetDims(), grad_expanded_x_tensor_expected_shape);

    auto grad_scales_tensor_desc = op.GetOutputDescByName("grad_scales");
    std::vector<int64_t> grad_scales_tensor_expected_shape = {5, 3};
    EXPECT_EQ(grad_scales_tensor_desc.GetShape().GetDims(), grad_scales_tensor_expected_shape);
}

TEST_F(MoeFinalizeRoutingV2GradProto, dtype_infer)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2Grad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRoutingV2Grad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref1 = ge::DT_FLOAT;
        ge::DataType input_ref2 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(6, 2)
                .IrInstanceNum({1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref1, &input_ref2, &input_ref1, &input_ref1, &input_ref2, &input_ref1})
                .OutputDataTypes({&output_ref, &output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
    }
}

TEST_F(MoeFinalizeRoutingV2GradProto, invalid_shape_infer_0)
{
    ge::op::MoeFinalizeRoutingV2Grad op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};

    auto grad_y_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {5, 8}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("grad_y", grad_y_tensor_desc);

    auto expanded_row_idx_tensor_desc =
        create_desc_shape_range({15}, ge::DT_INT32, ge::FORMAT_ND, {15}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);

    auto ret = InferShapeTest(op);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto grad_expanded_x_tensor_desc = op.GetOutputDescByName("grad_expanded_x");
    std::vector<int64_t> grad_expanded_x_tensor_expected_shape = {-2};
    EXPECT_EQ(grad_expanded_x_tensor_desc.GetShape().GetDims(), grad_expanded_x_tensor_expected_shape);

    auto grad_scales_tensor_desc = op.GetOutputDescByName("grad_scales");
    std::vector<int64_t> grad_scales_tensor_expected_shape = {-2};
    EXPECT_EQ(grad_scales_tensor_desc.GetShape().GetDims(), grad_scales_tensor_expected_shape);
}

TEST_F(MoeFinalizeRoutingV2GradProto, invalid_shape_infer_1)
{
    ge::op::MoeFinalizeRoutingV2Grad op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};

    auto grad_y_tensor_desc =
        create_desc_shape_range({15, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {5, 8}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("grad_y", grad_y_tensor_desc);

    auto expanded_row_idx_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_INT32, ge::FORMAT_ND, {15}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);

    auto ret = InferShapeTest(op);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto grad_expanded_x_tensor_desc = op.GetOutputDescByName("grad_expanded_x");
    std::vector<int64_t> grad_expanded_x_tensor_expected_shape = {-2};
    EXPECT_EQ(grad_expanded_x_tensor_desc.GetShape().GetDims(), grad_expanded_x_tensor_expected_shape);

    auto grad_scales_tensor_desc = op.GetOutputDescByName("grad_scales");
    std::vector<int64_t> grad_scales_tensor_expected_shape = {-2};
    EXPECT_EQ(grad_scales_tensor_desc.GetShape().GetDims(), grad_scales_tensor_expected_shape);
}

TEST_F(MoeFinalizeRoutingV2GradProto, shape_infer_option_param_shape_invalid)
{
    ge::op::MoeFinalizeRoutingV2Grad op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};

    auto grad_y_tensor_desc =
        create_desc_shape_range({5, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {5, 8}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("grad_y", grad_y_tensor_desc);

    auto expanded_row_idx_tensor_desc =
        create_desc_shape_range({15}, ge::DT_INT32, ge::FORMAT_ND, {15}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);

    auto expanded_x_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {15, 8}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_x_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {5, 3}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto ret = InferShapeTest(op);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeFinalizeRoutingV2GradProto, shape_infer_set_drop_mode_on)
{
    ge::op::MoeFinalizeRoutingV2Grad op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};

    auto grad_y_tensor_desc =
        create_desc_shape_range({5, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {5, 8}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("grad_y", grad_y_tensor_desc);

    auto expanded_row_idx_tensor_desc =
        create_desc_shape_range({15}, ge::DT_INT32, ge::FORMAT_ND, {15}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_row_idx_tensor_desc);

    auto expanded_x_tensor_desc =
        create_desc_shape_range({15, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {15, 8}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_x_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({5, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {5, 3}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expert_idx_tensor_desc =
        create_desc_shape_range({5, 3}, ge::DT_INT32, ge::FORMAT_ND, {5, 3}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expert_idx", expert_idx_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({8, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {8, 8}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);
    op.SetAttr("drop_pad_mode", 1);

    auto ret = InferShapeTest(op);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto grad_expanded_x_tensor_desc = op.GetOutputDescByName("grad_expanded_x");
    std::vector<int64_t> grad_expanded_x_tensor_expected_shape = {15, 8};
    EXPECT_EQ(grad_expanded_x_tensor_desc.GetShape().GetDims(), grad_expanded_x_tensor_expected_shape);

    auto grad_scales_tensor_desc = op.GetOutputDescByName("grad_scales");
    std::vector<int64_t> grad_scales_tensor_expected_shape = {5, 3};
    EXPECT_EQ(grad_scales_tensor_desc.GetShape().GetDims(), grad_scales_tensor_expected_shape);
}
