/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#pragma once
#include "kernel_operator.h"

namespace npu_ops_transformer_ext{
using namespace AscendC;

__aicore__ inline UnaryRepeatParams MakeDefaultUnaryRepeatParams()
{
    UnaryRepeatParams p;
    p.dstBlkStride = 1;
    p.srcBlkStride = 1;
    p.dstRepStride = 8;
    p.srcRepStride = 8;
    return p;
}

__aicore__ inline UnaryRepeatParams CastHalf2FloatRepeatParams()
{
    UnaryRepeatParams p;
    p.dstBlkStride = 1;
    p.srcBlkStride = 1;
    p.dstRepStride = 8;
    p.srcRepStride = 4;
    return p;
}

__aicore__ inline UnaryRepeatParams CastFloat2HalfRepeatParams()
{
    UnaryRepeatParams p;
    p.dstBlkStride = 1;
    p.srcBlkStride = 1;
    p.dstRepStride = 4;
    p.srcRepStride = 8;
    return p;
}

__aicore__ inline BinaryRepeatParams MakeDefaultBinaryRepeatParams()
{
    BinaryRepeatParams p;
    p.dstBlkStride = 1;
    p.src0BlkStride = 1;
    p.src1BlkStride = 1;
    p.dstRepStride = 8;
    p.src0RepStride = 8;
    p.src1RepStride = 8;
    return p;
}

}