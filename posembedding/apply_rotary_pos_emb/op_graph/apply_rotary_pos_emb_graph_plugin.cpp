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
 * \file apply_rotary_pos_emb_graph_plugin.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static ge::graphStatus ApplyRotaryPosEmbInferDtype(gert::InferDataTypeContext *context)
{
    OP_LOGD(context, "ApplyRotaryPosEmbInferDtype begin.");
    context->SetOutputDataType(OUTPUT_0_IDX, context->GetInputDataType(INPUT0));
    context->SetOutputDataType(OUTPUT_1_IDX, context->GetInputDataType(INPUT1));
    OP_LOGD(context, "ApplyRotaryPosEmbInferDtype end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ApplyRotaryPosEmb).InferDataType(ApplyRotaryPosEmbInferDtype);
}; // namespace ops
