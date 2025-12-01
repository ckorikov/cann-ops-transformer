/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lower_triangular_inverse.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LowerTriangularInverse);

const aclTensor *LowerTriangularInverse(const aclTensor *x, const aclTensor *y, aclOpExecutor *executor)
{
    L0_DFX(LowerTriangularInverse, x, y);
    DataType outType = DataType::DT_FLOAT; // 输出类型
    Format format = Format::FORMAT_ND; // 输出分形
    auto output = executor->AllocTensor(outType, format, format);

    auto ret = INFER_SHAPE(LowerTriangularInverse, OP_INPUT(x), OP_OUTPUT(y));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "LowerTriangularInverse InferShape failed.");
    auto ret1 = ADD_TO_LAUNCHER_LIST_AICORE(LowerTriangularInverse, OP_INPUT(x), OP_OUTPUT(y));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret1 != ACLNN_SUCCESS, return nullptr,
        "LowerTriangularInverse ADD_TO_LAUNCHER_LIST_AICORE failed.");
        
    return y;
}
}