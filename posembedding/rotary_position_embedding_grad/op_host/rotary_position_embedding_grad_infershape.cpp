/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file rotary_position_embedding_grad.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_GRAD_INDEX = 0;
static constexpr size_t INPUT_COS_INDEX = 1;
static constexpr size_t INPUT_SIN_INDEX = 2;
static constexpr size_t OUTPUT_DX_INDEX = 0;
static constexpr size_t OUTPUT_DCOS_INDEX = 1;
static constexpr size_t OUTPUT_DSIN_INDEX = 2;

static ge::graphStatus InferShapeForRotaryPositionEmbeddingGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForRotaryPositionEmbeddingGrad.");
    const gert::Shape* gradShape = context->GetInputShape(INPUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* cosShape = context->GetInputShape(INPUT_COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosShape);
    const gert::Shape* sinShape = context->GetInputShape(INPUT_SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinShape);
    gert::Shape* dxShape = context->GetOutputShape(OUTPUT_DX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);
    gert::Shape* dcosShape = context->GetOutputShape(OUTPUT_DCOS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dcosShape);
    gert::Shape* dsinShape = context->GetOutputShape(OUTPUT_DSIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dsinShape);

    *dxShape = *gradShape;
    *dcosShape = *cosShape;
    *dsinShape = *sinShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForRotaryPositionEmbeddingGrad.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForRotaryPositionEmbeddingGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_DX_INDEX, context->GetInputDataType(INPUT_GRAD_INDEX));
    context->SetOutputDataType(OUTPUT_DCOS_INDEX, context->GetInputDataType(INPUT_COS_INDEX));
    context->SetOutputDataType(OUTPUT_DSIN_INDEX, context->GetInputDataType(INPUT_SIN_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotaryPositionEmbeddingGrad)
    .InferShape(InferShapeForRotaryPositionEmbeddingGrad)
    .InferDataType(InferDataTypeForRotaryPositionEmbeddingGrad);
} // namespace ops