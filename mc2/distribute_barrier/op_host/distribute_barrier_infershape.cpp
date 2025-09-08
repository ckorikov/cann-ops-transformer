/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

/*!
 * \file distribute_barrier_infer.cc
 * \brief
 */
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
using namespace ge;
namespace ops {

static constexpr size_t DIM_ONE = 1UL;

static constexpr size_t BARRIER_INPUT_X_REF_INDEX = 0;
static constexpr size_t BARRIER_OUTPUT_X_REF_INDEX = 0;

static ge::graphStatus InferShapeDistributeBarrier(
    gert::InferShapeContext *context) {
  OPS_LOG_D(context->GetNodeName(), "Begin to do InferShapeDistributeBarrier.");
  // 获取输入shape
  const gert::Shape *xRefInputShape =
      context->GetInputShape(BARRIER_INPUT_X_REF_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, xRefInputShape);
  gert::Shape *xRefOutputShape =
      context->GetOutputShape(BARRIER_OUTPUT_X_REF_INDEX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, xRefOutputShape);

  // 这里要获取输入的dim，然后循环给输出赋值

  *xRefOutputShape = *xRefInputShape;
  OPS_LOG_D(context->GetNodeName(), "x_ref shape is :%s after infershape.",
            Ops::Base::ToString(*xRefOutputShape).c_str());
  OPS_LOG_D(context->GetNodeName(), "End to do InferShapeDistributeBarrier.");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDistributeBarrier(
    gert::InferDataTypeContext *context) {
  OPS_LOG_D(context->GetNodeName(),
            "Begin to do InferDataTypeDistributeBarrier.");
  auto xRefDtype = context->GetInputDataType(BARRIER_INPUT_X_REF_INDEX);
  context->SetOutputDataType(BARRIER_OUTPUT_X_REF_INDEX, xRefDtype);
  OPS_LOG_D(context->GetNodeName(),
            "End to do InferDataTypeDistributeBarrier.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DistributeBarrier)
    .InferShape(InferShapeDistributeBarrier)
    .InferDataType(InferDataTypeDistributeBarrier);
}  // namespace ops