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
 * \file all_gather_matmul_infershape.cc
 * \brief
 */
#include "mc2_log.h"
#include "context_util.h"
#include "register/op_impl_registry.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_common_infershape.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeAllGatherMatmul(gert::InferShapeContext* context)
{
    OP_LOGE_IF(
        InferShapeAllGatherMatmulCommon(context) != GRAPH_SUCCESS, GRAPH_FAILED, context->GetNodeName(),
        "infer shape excute failed.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAllGatherMatmul(gert::InferDataTypeContext* context) {
    auto d_type = context->GetInputDataType(0);
    context->SetOutputDataType(0, d_type);
    context->SetOutputDataType(1, d_type);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AllGatherMatmul).InferShape(InferShapeAllGatherMatmul).InferDataType(InferDataTypeAllGatherMatmul);
} // namespace ops
