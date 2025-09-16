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
 * \file moe_distribute_dispatch_infer.cc
 * \brief
 */
#include "runtime_util.h"
#include "mc2_log.h"
#include "platform/platform_info.h"
using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t RANK_NUM_PER_NODE = 8;


static constexpr size_t COMBINE_INPUT_EXPERT_X_INDEX = 0;
static constexpr size_t COMBINE_INPUT_EXPERT_IDX_INDEX = 1;
static constexpr size_t COMBINE_OUTPUT_X_INDEX = 0;


static ge::graphStatus InferShapeMoeDistributeCombine(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeCombine.");
    // 获取输入shape
    const gert::Shape *expandXShape = context->GetInputShape(COMBINE_INPUT_EXPERT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);
    const gert::Shape *expandIdsShape = context->GetInputShape(COMBINE_INPUT_EXPERT_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandIdsShape);
    gert::Shape *xShape = context->GetOutputShape(COMBINE_OUTPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

    int64_t bs = expandIdsShape->GetDimNum() == 1U ? NEG_ONE : expandIdsShape->GetDim(0);
    int64_t h = expandXShape->GetDimNum() == 1U ? NEG_ONE : expandXShape->GetDim(1);

    xShape->SetDimNum(DIM_TWO);
    xShape->SetDim(0U, bs);
    xShape->SetDim(1U, h);

    OP_LOGD(context->GetNodeName(), "x shape shape is :%s after infershape.",
        Ops::Base::ToString(*xShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeCombine.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeCombine(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeCombine.");
    auto xDtype = context->GetInputDataType(COMBINE_INPUT_EXPERT_X_INDEX);
    context->SetOutputDataType(COMBINE_OUTPUT_X_INDEX, xDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeCombine.");
    return ge::GRAPH_SUCCESS;
}



IMPL_OP_INFERSHAPE(MoeDistributeCombine)
    .InferShape(InferShapeMoeDistributeCombine)
    .InferDataType(InferDataTypeMoeDistributeCombine);

}  // namespace ops