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

class MoeFinalizeRoutingProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingProto TearDown" << std::endl;
    }
};

TEST_F(MoeFinalizeRoutingProto, normal_shape)
{
    ge::op::MoeFinalizeRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_expert_idx", expert_for_source_row_tensor_desc);

    auto ret = InferShapeTest(op);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto out_shape = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_output_shape = {3, 5};
    EXPECT_EQ(out_shape.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MoeFinalizeRoutingProto, neg_one_shape)
{
    ge::op::MoeFinalizeRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {-1, -1}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_expert_idx", expert_for_source_row_tensor_desc);

    auto ret = InferShapeTest(op);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto out_shape = op.GetOutputDescByName("y");
    std::vector<int64_t> expected_output_shape = {-1, -1};
    EXPECT_EQ(out_shape.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MoeFinalizeRoutingProto, error_shape)
{
    ge::op::MoeFinalizeRouting op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 1024}};
    auto expanded_permuted_rows_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_x", expanded_permuted_rows_tensor_desc);

    auto skip1_tensor_desc =
        create_desc_shape_range({3, 5, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5, 1}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", skip1_tensor_desc);

    auto skip2_tensor_desc =
        create_desc_shape_range({3, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x2", skip2_tensor_desc);

    auto bias_tensor_desc =
        create_desc_shape_range({6, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 5}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("bias", bias_tensor_desc);

    auto scales_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("scales", scales_tensor_desc);

    auto expanded_src_to_dst_row_tensor_desc =
        create_desc_shape_range({6}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_row_idx", expanded_src_to_dst_row_tensor_desc);

    auto expert_for_source_row_tensor_desc =
        create_desc_shape_range({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("expanded_expert_idx", expert_for_source_row_tensor_desc);

    auto ret = InferShapeTest(op);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MoeFinalizeRoutingProto, dtype_infer)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRouting"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeFinalizeRouting")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref1, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}
