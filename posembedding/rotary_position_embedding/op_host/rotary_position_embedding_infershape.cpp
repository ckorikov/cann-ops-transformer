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
 * \file rotary_position_embedding.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t INPUT_COS_INDEX = 1;
static constexpr size_t INPUT_SIN_INDEX = 2;
static constexpr size_t OUTPUT_Y_INDEX = 0;

namespace ops {
static ge::graphStatus InferShapeForRotaryPositionEmbedding(gert::InferShapeContext *context)
{
    OP_LOGD(context, "Begin to do InferShapeForRotaryPositionEmbedding.");
    const gert::Shape *xShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *cosShape = context->GetInputShape(INPUT_COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosShape);
    const gert::Shape *sinShape = context->GetInputShape(INPUT_SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinShape);
    gert::Shape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    *yShape = *xShape;

    OP_LOGD(context, "End to do InferShapeForRotaryPositionEmbedding.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForRotaryPositionEmbedding(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotaryPositionEmbedding)
    .InferShape(InferShapeForRotaryPositionEmbedding)
    .InferDataType(InferDataTypeForRotaryPositionEmbedding);
} // namespace ops
