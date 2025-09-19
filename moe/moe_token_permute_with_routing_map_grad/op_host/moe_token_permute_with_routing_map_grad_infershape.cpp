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
 * \file moe_token_permute_with_routing_map_grad_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForMoeTokenPermuteWithRoutingMapGrad(gert::InferShapeContext* context)
{
    const gert::Shape* permuted_token_output_grad_shape = context->GetInputShape(0);
    const int* tokens_num = context->GetAttrs()->GetAttrPointer<int>(1);
    int64_t tokensNum = static_cast<int64_t>(*tokens_num);

    int64_t hidden_size = permuted_token_output_grad_shape->GetDim(1);
    int64_t topk_num = permuted_token_output_grad_shape->GetDim(0) / tokensNum;

    gert::Shape* tokens_grad_out_shape = context->GetOutputShape(0);
    gert::Shape* probs_grad_out_optional_shape = context->GetOutputShape(1);
    const int64_t out_dim_num = 2;
    tokens_grad_out_shape->SetDimNum(out_dim_num);
    tokens_grad_out_shape->SetDim(0, tokensNum);
    tokens_grad_out_shape->SetDim(1, hidden_size);
    probs_grad_out_optional_shape->SetDimNum(out_dim_num);
    probs_grad_out_optional_shape->SetDim(0, tokensNum);
    probs_grad_out_optional_shape->SetDim(1, topk_num);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMoeTokenPermuteWithRoutingMapGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenPermuteWithRoutingMapGrad)
    .InferShape(InferShapeForMoeTokenPermuteWithRoutingMapGrad)
    .InferDataType(InferDataTypeForMoeTokenPermuteWithRoutingMapGrad);
} // namespace ops