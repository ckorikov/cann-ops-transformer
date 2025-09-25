/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <iostream>
 #include "gtest/gtest.h"
 #include "exe_graph/runtime/storage_format.h"
 #include "exe_graph/runtime/storage_shape.h"
 #include "register/op_impl_registry_base.h"
 #include "register/op_impl_registry.h"
 #include "kernel_run_context_facker.h"
 #include "op_proto_test_util.h"
 #include "nn_norm_ops.h"
 #include "common/utils/ut_op_common.h"
 #include "matrix_calculation_ops.h"
 #include "fusion_ops.h"
 #include "array_ops.h"
 #include "error_util.h"
 #include "util/util.h"
 
 class GroupedMatmulFinalizeRouting : public testing::Test {
 protected:
     static void SetUpTestCase() {
         std::cout << "GroupedMatmulFinalizeRouting Proto Test SetUp" << std::endl;
     }
 
     static void TearDownTestCase() {
         std::cout << "GroupedMatmulFinalizeRouting Proto Test TearDown" << std::endl;
     }
 };
 
 TEST_F(GroupedMatmulFinalizeRouting, grouped_matmul_finalize_routing_1) {
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bsdp = 128;
    int bs = 64;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape scaleShape = {{e, n}, {e, n}};
    gert::StorageShape pertoken_scaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bsdp, n}, {bsdp, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};
    gert::StorageShape yShape = {{bs, n}, {bs, n}};
 
    ge::op::GroupedMatmulFinalizeRouting op;
    op.UpdateInputDesc("x", create_desc({m, k}, ge::DT_INT8));
    op.UpdateInputDesc("w", create_desc({e, k, n}, ge::DT_INT8));
    op.UpdateInputDesc("scale", create_desc({e, n}, ge::DT_FLOAT));
    op.UpdateInputDesc("bias", create_desc({}, ge::DT_FLOAT));
    op.UpdateInputDesc("pertoken_scale", create_desc({m}, ge::DT_FLOAT));
    op.UpdateInputDesc("group_list", create_desc({e}, ge::DT_INT64));
    op.UpdateInputDesc("shared_input", create_desc({bsdp, n}, ge::DT_BF16));
    op.UpdateInputDesc("logit", create_desc({m}, ge::DT_FLOAT));
    op.UpdateInputDesc("row_index", create_desc({m}, ge::DT_INT64));
 
    op.SetAttr("dtype", 0);
    op.SetAttr("shared_input_weight", float(1.0));
    op.SetAttr("shared_input_offset", int(0));
    op.SetAttr("transpose_x", false);
    op.SetAttr("transpose_w", false);
    op.SetAttr("output_bs", bs);
    op.SetAttr("group_list_type", 1);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    auto inferShapeResult = InferShapeTest(op);
    std::cout << "GroupedMatmulFinalizeRouting ops_test, inferShapeResult: " << inferShapeResult << std::endl;
    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
    auto inferDataTypeResult = InferDataTypeTest(op);
    std::cout << "GroupedMatmulFinalizeRouting ops_test, inferDataTypeResult: " << inferDataTypeResult << std::endl;
 }

 TEST_F(GroupedMatmulFinalizeRouting, grouped_matmul_finalize_routing_2) {
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;
    int bsdp = 128;
    int bs = 64;
    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape scaleShape = {{e, n}, {e, n}};
    gert::StorageShape pertoken_scaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape shared_inputShape = {{bsdp, n}, {bsdp, n}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};
    gert::StorageShape yShape = {{bs, n}, {bs, n}};
 
    ge::op::GroupedMatmulFinalizeRouting op;
    op.UpdateInputDesc("x", create_desc({m, k}, ge::DT_INT8));
    op.UpdateInputDesc("w", create_desc({e, k, n}, ge::DT_INT8));
    op.UpdateInputDesc("scale", create_desc({e, n}, ge::DT_FLOAT));
    op.UpdateInputDesc("bias", create_desc({}, ge::DT_FLOAT));
    op.UpdateInputDesc("pertoken_scale", create_desc({m}, ge::DT_FLOAT));
    op.UpdateInputDesc("group_list", create_desc({e}, ge::DT_INT64));
    op.UpdateInputDesc("shared_input", create_desc({}, ge::DT_BF16));
    op.UpdateInputDesc("logit", create_desc({}, ge::DT_FLOAT));
    op.UpdateInputDesc("row_index", create_desc({m}, ge::DT_INT64));
 
    op.SetAttr("dtype", 0);
    op.SetAttr("shared_input_weight", float(1.0));
    op.SetAttr("shared_input_offset", int(0));
    op.SetAttr("transpose_x", false);
    op.SetAttr("transpose_w", false);
    op.SetAttr("output_bs", bs);
    op.SetAttr("group_list_type", 1);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    auto inferShapeResult = InferShapeTest(op);
    std::cout << "GroupedMatmulFinalizeRouting ops_test, inferShapeResult: " << inferShapeResult << std::endl;
    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
    auto inferDataTypeResult = InferDataTypeTest(op);
    std::cout << "GroupedMatmulFinalizeRouting ops_test, inferDataTypeResult: " << inferDataTypeResult << std::endl;
 }