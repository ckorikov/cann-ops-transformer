/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_all_reduce.h
 * \brief kernel内核接口声明
 */
#ifndef QUANT_ALL_REDUCE_H
#define QUANT_ALL_REDUCE_H

#include <kernel_operator.h>
#include <kernel_tiling/kernel_tiling.h>
#include "quant_all_reduce_tiling_data.h"

namespace QuantAllReduceImpl {

using namespace AscendC;

// 自定义
#define TemplateMC2TypeClass bool Param1, bool Param2, bool Param3
#define TemplateMC2TypeFunc Param1, Param2, Param3

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template <TemplateMC2TypeClass>
class QuantAllReduce {
public:
    __aicore__ inline QuantAllReduce(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output, GM_ADDR workspaceGM, TPipe *pipe,
                                const QuantAllReduceTilingData *tilingData);
    __aicore__ inline void Process();

private:
};

template <TemplateMC2TypeClass>
__aicore__ inline void QuantAllReduce<TemplateMC2TypeFunc>::Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
                                                                 GM_ADDR workspaceGM, TPipe *pipe,
                                                                 const QuantAllReduceTilingData *tilingData)
{
}

template <TemplateMC2TypeClass>
__aicore__ inline void QuantAllReduce<TemplateMC2TypeFunc>::Process()
{
}

} // namespace QuantAllReduceImpl

#endif // QUANT_ALL_REDUCE_H
