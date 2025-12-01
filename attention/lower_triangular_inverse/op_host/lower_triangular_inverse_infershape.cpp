/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file chunk_gated_delta_rule_inverse.cc
 * \brief
 */
#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace gert;
using namespace ge;

namespace ops {

const int64_t X_INDEX = 0;
const int64_t DIM_LEN = 5;

static ge::graphStatus InferShapeLowerTriangularInverse(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeLowerTriangularInverse");
    const gert::Shape *xShape = context->GetDynamicInputShape(X_INDEX, 0);
    
    if (xShape->GetDimNum() != DIM_LEN) {
        OP_LOGE(context->GetNodeName(), "x shape is not equal to 5, actual is %d", xShape->GetDimNum());
    }

    auto outShape = context->GetOutputShape(0);
    outShape->SetDimNum(DIM_LEN);
    for (int i = 0; i < DIM_LEN; i++) {
        outShape->SetDim(i, xShape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeLowerTriangularInverse");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4LowerTriangularInverse(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, DataType::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LowerTriangularInverse)
    .InferShape(InferShapeLowerTriangularInverse)
    .InferDataType(InferDataType4LowerTriangularInverse);
} // namespace ops