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
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTokenPermuteGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenPermuteGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenPermuteGrad Proto Test TearDown" << std::endl;
  }
};

struct MoeTokenPermuteGradInfo 
{
    gert::StorageShape& permutedOutputGradShape;
    gert::StorageShape& sortedIndicesShape;
    std::vector<int64_t> expectOutShape;

    ge::DataType permutedOutputGradDtype;
    ge::DataType sortedIndicesDtype;

    ge::DataType yDtype;

    int64_t num_topk = 0;
    bool padded_mode = 0;
};

static std::vector<int64_t> ToVector(const gert::Shape& shape) {
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(const MoeTokenPermuteGradInfo ioInfo,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    gert::StorageShape yStorageShape = {};
    std::vector<gert::StorageShape*> ouputShapes = {&yStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("MoeTokenPermuteGrad")
        .NodeIoNum(2, 1)
        .NodeInputTd(0, ioInfo.permutedOutputGradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ioInfo.sortedIndicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)

        .NodeOutputTd(0, ioInfo.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        
        .InputTensors({(gert::Tensor *)&ioInfo.permutedOutputGradShape})
        .InputTensors({(gert::Tensor *)&ioInfo.sortedIndicesShape})


        .OutputShapes(ouputShapes)
        .Attr("num_topk", int64_t(ioInfo.num_topk))
        .Attr("padded_mode", bool(ioInfo.padded_mode))
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MoeTokenPermuteGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    EXPECT_EQ(ToVector(yStorageShape.GetOriginShape()), ioInfo.expectOutShape);
}

TEST_F(MoeTokenPermuteGrad, infershape_bf16)
{
    gert::StorageShape permutedOutputGradShape = {{49152, 5120}, {49152, 5120}};
    gert::StorageShape sortedIndicesShape = {{49152}, {49152}};

    std::vector<int64_t> expectOutShape = {6144, 5120};
    MoeTokenPermuteGradInfo ioInfoT = {permutedOutputGradShape, sortedIndicesShape, expectOutShape,
    ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, 8, false};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeTokenPermuteGrad, infershape_fp16)
{
    gert::StorageShape permutedOutputGradShape = {{49152, 5120}, {49152, 5120}};
    gert::StorageShape sortedIndicesShape = {{49152}, {49152}};

    std::vector<int64_t> expectOutShape = {6144, 5120};
    MoeTokenPermuteGradInfo ioInfoT = {permutedOutputGradShape, sortedIndicesShape, expectOutShape,
    ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, 8, false};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeTokenPermuteGrad, infershape_fp32)
{
    gert::StorageShape permutedOutputGradShape = {{49152, 5120}, {49152, 5120}};
    gert::StorageShape sortedIndicesShape = {{49152}, {49152}};

    std::vector<int64_t> expectOutShape = {6144, 5120};
    MoeTokenPermuteGradInfo ioInfoT = {permutedOutputGradShape, sortedIndicesShape, expectOutShape,
    ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 8, false};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}
