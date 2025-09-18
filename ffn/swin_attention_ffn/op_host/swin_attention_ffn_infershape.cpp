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
 * \file swin_attention_ffn.cc
 * \brief
 */
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops
{
static ge::graphStatus InferShapeSwinAttentionFFN(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter SwinAttentionFFN infershape impl.");

    const gert::Shape* x1_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);

    gert::Shape* y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    *y_shape = *x1_shape;
    OP_LOGI(context->GetNodeName(), "SwinAttentionFFN infershape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSwinAttentionFFN(gert::InferDataTypeContext *context)
{
    OP_LOGI(context->GetNodeName(), "Enter SwinAttentionFFN inferDataType impl.");

    const ge::DataType x1_data_type = context->GetInputDataType(0);
    ge::graphStatus ret = context->SetOutputDataType(0, x1_data_type);

    OP_LOGI(context->GetNodeName(), "SwinAttentionFFN inferDataType end.");
    return ret;
}
    
IMPL_OP_INFERSHAPE(SwinAttentionFFN).InferShape(InferShapeSwinAttentionFFN).InferDataType(InferDataTypeSwinAttentionFFN);
}   // namespace ops