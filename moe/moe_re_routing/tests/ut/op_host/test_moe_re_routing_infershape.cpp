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
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "experiment_ops.h"
#include "fusion_ops.h"

class MoeReRouting : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeReRouting Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeReRouting Proto Test TearDown" << std::endl;
    }
};

TEST_F(MoeReRouting, MoeReRouting_infershape)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeReRouting"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeReRouting")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape tokens_shape = {{256, 7168}, {256, 7168}};
    gert::StorageShape expert_token_num_per_rank_shape = {{16, 16}, {16, 16}};
    gert::StorageShape per_token_scales_shape = {{256}, {256}};
    gert::StorageShape permute_tokens_shape_out = {{256, 7168}, {256, 7168}};
    gert::StorageShape permute_tokens_scales_shape_out = {{256}, {256}};
    gert::StorageShape permute_token_idx_shape_out = {{256}, {256}};
    gert::StorageShape expert_token_num_shape_out = {{16}, {16}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 4)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&tokens_shape, &expert_token_num_per_rank_shape, &per_token_scales_shape})
                      .OutputShapes({&permute_tokens_shape_out,
                          &permute_tokens_scales_shape_out,
                          &permute_token_idx_shape_out,
                          &expert_token_num_shape_out})
                      .NodeAttrs({})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto output2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    auto output3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);

    ASSERT_EQ(ge::Shape2String(*output0), "[256, 7168]");
    ASSERT_EQ(ge::Shape2String(*output1), "[256]");
    ASSERT_EQ(ge::Shape2String(*output2), "[256]");
    ASSERT_EQ(ge::Shape2String(*output3), "[16]");
}

TEST_F(MoeReRouting, MoeReRouting_infershape_scale)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeReRouting"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeReRouting")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape tokens_shape = {{36, 385}, {36, 385}};
    gert::StorageShape expert_token_num_per_rank_shape = {{3, 12}, {3, 12}};
    gert::StorageShape per_token_scales_shape = {{36, 3}, {36, 3}};
    gert::StorageShape permute_tokens_shape_out = {{36, 385}, {36, 385}};
    gert::StorageShape permute_tokens_scales_shape_out = {{36, 3}, {36, 3}};
    gert::StorageShape permute_token_idx_shape_out = {{36}, {36}};
    gert::StorageShape expert_token_num_shape_out = {{12}, {12}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 4)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&tokens_shape, &expert_token_num_per_rank_shape, &per_token_scales_shape})
                      .OutputShapes({&permute_tokens_shape_out,
                          &permute_tokens_scales_shape_out,
                          &permute_token_idx_shape_out,
                          &expert_token_num_shape_out})
                      .NodeAttrs({})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto output2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    auto output3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);

    ASSERT_EQ(ge::Shape2String(*output0), "[36, 385]");
    ASSERT_EQ(ge::Shape2String(*output1), "[36, 3]");
    ASSERT_EQ(ge::Shape2String(*output2), "[36]");
    ASSERT_EQ(ge::Shape2String(*output3), "[12]");
}

TEST_F(MoeReRouting, MoeReRouting_infershape_dynamic)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeReRouting"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeReRouting")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape tokens_shape = {{-1, -1}, {-1, -1}};
    gert::StorageShape expert_token_num_per_rank_shape = {{-1, -1}, {-1, -1}};
    gert::StorageShape per_token_scales_shape = {{-1, -1}, {-1, -1}};
    gert::StorageShape permute_tokens_shape_out = {{-1, -1}, {-1, -1}};
    gert::StorageShape permute_tokens_scales_shape_out = {{-1, -1}, {-1, -1}};
    gert::StorageShape permute_token_idx_shape_out = {{-1}, {-1}};
    gert::StorageShape expert_token_num_shape_out = {{-1}, {-1}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 4)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&tokens_shape, &expert_token_num_per_rank_shape, &per_token_scales_shape})
                      .OutputShapes({&permute_tokens_shape_out,
                          &permute_tokens_scales_shape_out,
                          &permute_token_idx_shape_out,
                          &expert_token_num_shape_out})
                      .NodeAttrs({})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto output2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    auto output3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);

    ASSERT_EQ(ge::Shape2String(*output0), "[-1, -1]");
    ASSERT_EQ(ge::Shape2String(*output1), "[-1, -1]");
    ASSERT_EQ(ge::Shape2String(*output2), "[-1]");
    ASSERT_EQ(ge::Shape2String(*output3), "[-1]");
}

TEST_F(MoeReRouting, MoeReRouting_infershape_unkownshape)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeReRouting"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeReRouting")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape tokens_shape = {{-2}, {-2}};
    gert::StorageShape expert_token_num_per_rank_shape = {{-2}, {-2}};
    gert::StorageShape per_token_scales_shape = {{-2}, {-2}};
    gert::StorageShape permute_tokens_shape_out = {{-2}, {-2}};
    gert::StorageShape permute_tokens_scales_shape_out = {{-2}, {-2}};
    gert::StorageShape permute_token_idx_shape_out = {{-2}, {-2}};
    gert::StorageShape expert_token_num_shape_out = {{-2}, {-2}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 4)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&tokens_shape, &expert_token_num_per_rank_shape, &per_token_scales_shape})
                      .OutputShapes({&permute_tokens_shape_out,
                          &permute_tokens_scales_shape_out,
                          &permute_token_idx_shape_out,
                          &expert_token_num_shape_out})
                      .NodeAttrs({})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    auto output2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    auto output3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);

    ASSERT_EQ(ge::Shape2String(*output0), "[-2]");
    ASSERT_EQ(ge::Shape2String(*output1), "[-2]");
    ASSERT_EQ(ge::Shape2String(*output2), "[-2]");
    ASSERT_EQ(ge::Shape2String(*output3), "[-2]");
}

TEST_F(MoeReRouting, MoeReRouting_inferdtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeReRouting"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeReRouting")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType input_1 = ge::DT_INT32;
    ge::DataType input_2 = ge::DT_FLOAT;
    ge::DataType output_0 = ge::DT_FLOAT16;
    ge::DataType output_1 = ge::DT_FLOAT;
    ge::DataType output_2 = ge::DT_INT32;
    ge::DataType output_3 = ge::DT_INT32;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 4)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({})
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_0, &input_1, &input_2})
                              .OutputDataTypes({&output_0, &output_1, &output_2, &output_3})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_INT32);
}
