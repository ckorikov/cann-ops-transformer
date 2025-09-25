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
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

class DistributeBarrierRuntimeProtoTest : public testing::Test {
};

namespace
{
    bool Comp(const gert::Shape *x, const gert::Shape y) {
        for (int i = 0; i < x->GetDimNum(); i++) {
            std::cout << x->GetDim(i) << " " << y.GetDim(i) << std::endl;
            if (x->GetDim(i) != y.GetDim(i)) {
                return false;
            }
        }
        return true;
    }
} // namespace

// infer shape with bias, success
TEST_F(DistributeBarrierRuntimeProtoTest, infer_shape_0) {
    gert::StorageShape x_ref = {{32, 7168}, {32, 7168}};

    gert::StorageShape output = {{32, 7168}, {32, 7168}};

    std::string opType("DistributeBarrier");
    std::string group("group");
    int64_t world_size = 288;
    auto holder = gert::InferShapeContextFaker()
                        .NodeIoNum(1, 1)
                        .SetOpType(opType)
                        .IrInstanceNum({1})
                        .InputShapes({&x_ref})
                        .OutputShapes({&output})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"world_size", ge::AnyValue::CreateFrom<int64_t>(world_size)}})
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->infer_shape;
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
}