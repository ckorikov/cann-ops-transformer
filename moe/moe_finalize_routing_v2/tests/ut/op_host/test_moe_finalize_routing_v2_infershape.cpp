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

class MoeFinalizeRoutingV2Infershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2Proto TearDown" << std::endl;
    }
};

struct MoeFinalizeRoutingV2Info 
{
    gert::StorageShape& expandedXShape;
    gert::StorageShape& expandedRowIdxShape;
    gert::StorageShape& x1Shape;
    gert::StorageShape& x2Shape;
    gert::StorageShape& biasShape;
    gert::StorageShape& scalesShape;
    gert::StorageShape& expertIdxShape;
    std::vector<int64_t> expectOutShape;

    ge::DataType expandedXDtype;
    ge::DataType expandedRowIdxDtype;
    ge::DataType x1Dtype;
    ge::DataType x2Dtype;
    ge::DataType biasDtype;
    ge::DataType scalesDtype;
    ge::DataType expertIdxDtype;
    ge::DataType yDtype;

    int64_t dropPadMode = 0;
};

static std::vector<int64_t> ToVector(const gert::Shape& shape) {
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(const MoeFinalizeRoutingV2Info ioInfo,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    gert::StorageShape yStorageShape = {};
    std::vector<gert::StorageShape*> ouputShapes = {&yStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("MoeFinalizeRoutingV2")
        .NodeIoNum(7, 1)
        .NodeInputTd(0, ioInfo.expandedXDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ioInfo.expandedRowIdxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ioInfo.x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ioInfo.x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ioInfo.biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ioInfo.scalesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ioInfo.expertIdxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ioInfo.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        
        .InputTensors({(gert::Tensor *)&ioInfo.expandedXShape})
        .InputTensors({(gert::Tensor *)&ioInfo.expandedRowIdxShape})
        .InputTensors({(gert::Tensor *)&ioInfo.x1Shape})
        .InputTensors({(gert::Tensor *)&ioInfo.x2Shape})
        .InputTensors({(gert::Tensor *)&ioInfo.biasShape})
        .InputTensors({(gert::Tensor *)&ioInfo.scalesShape})
        .InputTensors({(gert::Tensor *)&ioInfo.expertIdxShape})

        .OutputShapes(ouputShapes)
        .Attr("drop_pad_mode", int64_t(ioInfo.dropPadMode))
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MoeFinalizeRoutingV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    EXPECT_EQ(ToVector(yStorageShape.GetOriginShape()), ioInfo.expectOutShape);
}

TEST_F(MoeFinalizeRoutingV2Infershape, moe_finalize_routing_v2_infershape_0)
{
    gert::StorageShape expandedXShape = {{6, 5}, {6, 5}};
    gert::StorageShape expandedRowIdxShape = {{6}, {6}};
    gert::StorageShape x1Shape = {{3, 5}, {3, 5}};
    gert::StorageShape x2Shape = {{3, 5}, {3, 5}};
    gert::StorageShape biasShape = {{6, 5}, {6, 5}};
    gert::StorageShape scalesShape = {{3, 2}, {3, 2}};
    gert::StorageShape expertIdxShape = {{3, 2}, {3, 2}};

    std::vector<int64_t> expectOutShape = {3, 5}; // scale第0维，expandedX第一维
    MoeFinalizeRoutingV2Info ioInfoT = {expandedXShape, expandedRowIdxShape, x1Shape, x2Shape, biasShape, scalesShape, expertIdxShape, expectOutShape,
    ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}