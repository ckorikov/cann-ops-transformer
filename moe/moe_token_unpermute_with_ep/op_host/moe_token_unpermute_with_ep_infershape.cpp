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
 * \file moe_token_unpermute_with_ep_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
using namespace ge;
namespace ops {

static constexpr int64_t UNPERMUTE_WITH_EP_INPUT_TOKENS = 0;
static constexpr int64_t UNPERMUTE_WITH_EP_INPUT_IDX = 1;
static constexpr int64_t UNPERMUTE_WITH_EP_INPUT_PROBS = 2;
static constexpr int64_t UNPERMUTE_WITH_EP_OUTPUT_TOKENS = 0;
static constexpr int64_t UNPERMUTE_WITH_EP_ARRT_TOPK = 0;
static constexpr int64_t UNPERMUTE_WITH_EP_ARRT_RANGE = 1;

static ge::graphStatus InferShapeForMoeTokenUnpermuteWithEp(gert::InferShapeContext* context)
{
    const gert::Shape* permuted_inputs_shape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_TOKENS);
    const gert::Shape* probs_shape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_PROBS);
    const int* topk = context->GetAttrs()->GetAttrPointer<int>(UNPERMUTE_WITH_EP_ARRT_TOPK);
    int64_t inputTopK = static_cast<int64_t>(*topk);
    int64_t tokens_num;
    if (probs_shape == nullptr) {
        const gert::Shape* indices_shape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_IDX);
        tokens_num = indices_shape->GetDim(0) / inputTopK;
    } else {
        tokens_num = probs_shape->GetDim(0);
    }

    gert::Shape* out_shape = context->GetOutputShape(UNPERMUTE_WITH_EP_OUTPUT_TOKENS);
    const int8_t out_dim_num = 2;
    out_shape->SetDimNum(out_dim_num);
    out_shape->SetDim(0, tokens_num);
    out_shape->SetDim(1, permuted_inputs_shape->GetDim(1));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMoeTokenUnpermuteWithEp(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(UNPERMUTE_WITH_EP_INPUT_TOKENS));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenUnpermuteWithEp)
    .InferShape(InferShapeForMoeTokenUnpermuteWithEp)
    .InferDataType(InferDataTypeForMoeTokenUnpermuteWithEp);
} // namespace ops
