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
#include "common/utils/ut_op_common.h"

class MoeTokenPermuteWithRoutingMapGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenPermuteWithRoutingMapGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenPermuteWithRoutingMapGrad Proto Test TearDown" << std::endl;
  }
};

TEST_F(MoeTokenPermuteWithRoutingMapGrad, MoeTokenPermuteWithRoutingMapGrad_infershape_bf16) {
  gert::StorageShape permutedTokenOutPutGradShape = {{1024, 7168}, {1024, 7168}};
  gert::StorageShape permutedProbsOutPutGradOptionalShape = {{1024}, {1024}};
  gert::StorageShape sortedIndicesShape = {{1024}, {1024}};
  gert::StorageShape routingMapOptionalShape = {{512, 512}, {512, 512}};
  // output
  gert::StorageShape tokensGradOutShape = {{512, 7168}, {512, 7168}};
  gert::StorageShape probsGradOutOptionalShape = {{512, 2}, {512, 2}};
  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("MoeTokenPermuteWithRoutingMapGrad")
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&permutedTokenOutPutGradShape, &permutedProbsOutPutGradOptionalShape, &sortedIndicesShape, &routingMapOptionalShape})
                    .OutputShapes({&tokensGradOutShape, &probsGradOutOptionalShape})
                    .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"num_experts", ge::AnyValue::CreateFrom<int64_t>(512)},
                      {"num_topk", ge::AnyValue::CreateFrom<int64_t>(512)},
                      {"padded_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeTokenPermuteWithRoutingMapGrad")->infer_shape;
  ge::graphStatus ret = infer_shape_func(context);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expectedTokensGradOutShape = {512, 7168};
  std::vector<int64_t> expectedProbsGradOutOptionalShape = {512, 2};
  auto TokensGradOutShape = context->GetOutputShape(0);
  auto ProbsGradOutOptionalShape = context->GetOutputShape(1);
  EXPECT_EQ(ops::ToVector(*TokensGradOutShape), expectedTokensGradOutShape);
  EXPECT_EQ(ops::ToVector(*ProbsGradOutOptionalShape), expectedProbsGradOutOptionalShape);
}
