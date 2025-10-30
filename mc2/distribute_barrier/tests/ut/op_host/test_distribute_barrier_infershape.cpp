/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class DistributeBarrierInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "DistributeBarrierInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DistributeBarrierInfershape TearDown" << std::endl;
    }
};

TEST_F(DistributeBarrierInfershape, infer_shape_0) {
    gert::StorageShape xStorageShape = {{32, 7168}, {32, 7168}};
    gert::StorageShape yStorageShape = {{32, 7168}, {32, 7168}};

    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {(gert::Tensor *)&xStorageShape};
    std::vector<gert::StorageShape*> ouputShapes = {&yStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("DistributeBarrier")
        .NodeIoNum(1, 1)
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(ouputShapes)
        .Attr("group", ge::AscendString("group"))
        .Attr("world_size", int64_t(288))
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("DistributeBarrier")->infer_shape;

    /* do infershape */
    ASSERT_EQ(inferShapeFunc(contextHolder.GetContext()), ge::GRAPH_SUCCESS);
}