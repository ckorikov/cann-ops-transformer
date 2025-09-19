/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved. reserved.
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
