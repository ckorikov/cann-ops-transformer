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
#include "common/utils/ut_op_common.h"
#include "experiment_ops.h"
#include "fusion_ops.h"

class InterleaveRope : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "InterleaveRope Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InterleaveRope Proto Test TearDown" << std::endl;
    }
};

TEST_F(InterleaveRope, InterleaveRope_infershape_b11d)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("InterleaveRope"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InterleaveRope")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape x_shape = {{32, 32, 1, 64}, {32, 32, 1, 64}};
    gert::StorageShape cos_shape = {{32, 1, 1, 64}, {32, 1, 1, 64}};
    gert::StorageShape sin_shape = {{32, 1, 1, 64}, {32, 1, 1, 64}};
    gert::StorageShape y_shape = {{32, 32, 1, 64}, {32, 32, 1, 64}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &cos_shape, &sin_shape})
                      .OutputShapes({&y_shape})
                      .NodeAttrs({})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output0), "[32, 32, 1, 64]");
}

TEST_F(InterleaveRope, InterleaveRope_infershape_b1sd)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("InterleaveRope"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InterleaveRope")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape x_shape = {{32, 32, 4, 64}, {32, 32, 4, 64}};
    gert::StorageShape cos_shape = {{32, 1, 4, 64}, {32, 1, 4, 64}};
    gert::StorageShape sin_shape = {{32, 1, 4, 64}, {32, 1, 4, 64}};
    gert::StorageShape y_shape = {{32, 32, 4, 64}, {32, 32, 4, 64}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &cos_shape, &sin_shape})
                      .OutputShapes({&y_shape})
                      .NodeAttrs({})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(ge::Shape2String(*output0), "[32, 32, 4, 64]");
}

TEST_F(InterleaveRope, InterleaveRope_inferdtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("InterleaveRope"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("InterleaveRope")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType output_0 = ge::DT_FLOAT16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({})
                              .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_0, &input_0, &input_0})
                              .OutputDataTypes({&output_0})
                              .Build();

    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

