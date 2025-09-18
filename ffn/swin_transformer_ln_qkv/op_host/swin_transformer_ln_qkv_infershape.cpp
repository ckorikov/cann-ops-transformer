/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file swin_transformer_ln_qkv.cc
 * \brief
 */
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeSwinTransformerLnQKV(gert::InferShapeContext* context) {
  OP_LOGI(context->GetNodeName(), "Enter SwinTransformerLnQKV infershape impl.");
  // x1_shape shape : (B, S, H)
  const gert::Shape* x1_shape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);

  gert::Shape* q_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, q_shape);
  gert::Shape* k_shape = context->GetOutputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, k_shape);
  gert::Shape* v_shape = context->GetOutputShape(2);
  OP_CHECK_NULL_WITH_CONTEXT(context, v_shape);
  uint32_t inputSize = 1;
  for (uint32_t dimIdx = 0; dimIdx < x1_shape->GetDimNum(); dimIdx++) {
      inputSize *= x1_shape->GetDim(dimIdx);
  }
  const uint32_t outputChannels = 4;
  OP_LOGW(context->GetNodeName(), "inputSize: %d",inputSize);
  q_shape->SetDimNum(outputChannels);
  k_shape->SetDimNum(outputChannels);
  v_shape->SetDimNum(outputChannels);
  const uint32_t outputChannel1 = 4;
  const uint32_t outputChannel2 = 64;
  const uint32_t outputChannel3 = 32;
  const uint32_t outputDim0 = 0;
  const uint32_t outputDim1 = 1;
  const uint32_t outputDim2 = 2;
  const uint32_t outputDim3 = 3;
  q_shape->SetDim(outputDim0, inputSize / (outputChannel1 * outputChannel2 * outputChannel3));
  q_shape->SetDim(outputDim1, outputChannel1);
  q_shape->SetDim(outputDim2, outputChannel2);
  q_shape->SetDim(outputDim3, outputChannel3);
  *k_shape = *q_shape;
  *v_shape = *q_shape;
  OP_LOGI(context->GetNodeName(), "SwinTransformerLnQKV infershape end.");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSwinTransformerLnQKV(gert::InferDataTypeContext *context) {
  OP_LOGI(context->GetNodeName(), "Enter SwinTransformerLnQKV inferDataType impl.");
  const ge::DataType dtype = context->GetInputDataType(0);
  // q_out, outidx:0
  ge::graphStatus ret0 = context->SetOutputDataType(0, dtype);
  // k_out, outidx:1
  context->SetOutputDataType(1, dtype);
  // v_out, outidx:2
  context->SetOutputDataType(2, dtype);
  OP_LOGI(context->GetNodeName(), "SwinTransformerLnQKV inferDataType end.");
  return ret0;
}

IMPL_OP_INFERSHAPE(SwinTransformerLnQKV).InferShape(InferShapeSwinTransformerLnQKV).InferDataType(InferDataTypeSwinTransformerLnQKV);

}  // namespace ops