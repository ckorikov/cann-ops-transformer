/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#include <iostream>
#include <gtest/gtest.h>
#include "common/utils/ut_op_common.h"

class RopeWithSinCosCache : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RopeWithSinCosCache Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RopeWithSinCosCache Proto Test TearDown" << std::endl;
    }
};

TEST_F(RopeWithSinCosCache, rope_with_sin_cos_cache_bf16_true)
{
    gert::StorageShape positionShape = {{48}, {48}};
    gert::StorageShape queryShape = {{48, 256}, {48, 256}};
    gert::StorageShape keyShape = {{48, 512}, {48, 512}};
    gert::StorageShape cosSinCacheShape = {{48, 128}, {48, 128}};
    vector<int64_t> mropeParams{0, 0, 0};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("RopeWithSinCosCache")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&positionShape, &queryShape, &keyShape, &cosSinCacheShape})
                      .OutputShapes({&queryShape, &keyShape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"numQHeads", ge::AnyValue::CreateFrom<int64_t>(2)},
                           {"numKHeads", ge::AnyValue::CreateFrom<int64_t>(4)},
                           {"headSize", ge::AnyValue::CreateFrom<int64_t>(128)},
                           {"mropeSection", ge::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
                           {"qstride", ge::AnyValue::CreateFrom<int64_t>(256)},
                           {"kstride", ge::AnyValue::CreateFrom<int64_t>(512)},
                           {"isNeoxStyle", ge::AnyValue::CreateFrom<int64_t>(1)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RopeWithSinCosCache")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedQueryOutputShape = {48, 256};
    std::vector<int64_t> expectedKeyOutputShape = {48, 512};
    auto queryOutShape = context->GetOutputShape(0);
    auto keyOutShape = context->GetOutputShape(1);
    EXPECT_EQ(ops::ToVector(*queryOutShape), expectedQueryOutputShape);
    EXPECT_EQ(ops::ToVector(*keyOutShape), expectedKeyOutputShape);
}

TEST_F(RopeWithSinCosCache, rope_with_sin_cos_cache_fp16_true)
{
    gert::StorageShape positionShape = {{48}, {48}};
    gert::StorageShape queryShape = {{48, 256}, {48, 256}};
    gert::StorageShape keyShape = {{48, 512}, {48, 512}};
    gert::StorageShape cosSinCacheShape = {{48, 128}, {48, 128}};
    vector<int64_t> mropeParams{0, 0, 0};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("RopeWithSinCosCache")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&positionShape, &queryShape, &keyShape, &cosSinCacheShape})
                      .OutputShapes({&queryShape, &keyShape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"numQHeads", ge::AnyValue::CreateFrom<int64_t>(2)},
                           {"numKHeads", ge::AnyValue::CreateFrom<int64_t>(4)},
                           {"headSize", ge::AnyValue::CreateFrom<int64_t>(128)},
                           {"mropeSection", ge::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
                           {"qstride", ge::AnyValue::CreateFrom<int64_t>(256)},
                           {"kstride", ge::AnyValue::CreateFrom<int64_t>(512)},
                           {"isNeoxStyle", ge::AnyValue::CreateFrom<int64_t>(1)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RopeWithSinCosCache")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedQueryOutputShape = {48, 256};
    std::vector<int64_t> expectedKeyOutputShape = {48, 512};
    auto queryOutShape = context->GetOutputShape(0);
    auto keyOutShape = context->GetOutputShape(1);
    EXPECT_EQ(ops::ToVector(*queryOutShape), expectedQueryOutputShape);
    EXPECT_EQ(ops::ToVector(*keyOutShape), expectedKeyOutputShape);
}

TEST_F(RopeWithSinCosCache, rope_with_sin_cos_cache_fp32_true)
{
    gert::StorageShape positionShape = {{48}, {48}};
    gert::StorageShape queryShape = {{48, 256}, {48, 256}};
    gert::StorageShape keyShape = {{48, 512}, {48, 512}};
    gert::StorageShape cosSinCacheShape = {{48, 128}, {48, 128}};
    vector<int64_t> mropeParams{0, 0, 0};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("RopeWithSinCosCache")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&positionShape, &queryShape, &keyShape, &cosSinCacheShape})
                      .OutputShapes({&queryShape, &keyShape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"numQHeads", ge::AnyValue::CreateFrom<int64_t>(2)},
                           {"numKHeads", ge::AnyValue::CreateFrom<int64_t>(4)},
                           {"headSize", ge::AnyValue::CreateFrom<int64_t>(128)},
                           {"mropeSection", ge::AnyValue::CreateFrom<vector<int64_t>>(mropeParams)},
                           {"qstride", ge::AnyValue::CreateFrom<int64_t>(256)},
                           {"kstride", ge::AnyValue::CreateFrom<int64_t>(512)},
                           {"isNeoxStyle", ge::AnyValue::CreateFrom<int64_t>(1)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RopeWithSinCosCache")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedQueryOutputShape = {48, 256};
    std::vector<int64_t> expectedKeyOutputShape = {48, 512};
    auto queryOutShape = context->GetOutputShape(0);
    auto keyOutShape = context->GetOutputShape(1);
    EXPECT_EQ(ops::ToVector(*queryOutShape), expectedQueryOutputShape);
    EXPECT_EQ(ops::ToVector(*keyOutShape), expectedKeyOutputShape);
}