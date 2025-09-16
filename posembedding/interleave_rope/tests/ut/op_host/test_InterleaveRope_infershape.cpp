/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

