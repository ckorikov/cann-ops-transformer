/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file check_winsize.h
 * \brief
 */

#ifndef CHECK_WINSIZE_H
#define CHECK_WINSIZE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_base.h"
#include "moe_distribute_dispatch_tiling.h"

__aicore__ inline void CheckWindowSize(
    uint64_t tilingWinSizeBytes, uint64_t realWinSizeBytes, AscendC::TPipe* tpipe_, GM_ADDR exceptionAddr)
{
    if (unlikely(realWinSizeBytes < tilingWinSizeBytes)) {
        constexpr uint64_t DATA_SIZE = 256; // 定义数据大小为256字节
        AscendC::GlobalTensor<int32_t> exceptionGlobal;
        exceptionGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(exceptionAddr), DATA_SIZE);
        AscendC::TBuf<AscendC::TPosition::VECCALC> exceptionBuf;
        tpipe_->InitBuffer(exceptionBuf, 1); // 初始化一个缓冲区
        AscendC::LocalTensor<int32_t> exceptionLocal = exceptionBuf.Get<int32_t>();
        AscendC::DataCopy(exceptionLocal[1], exceptionGlobal, 1); // 从全局地址复制数据到本地地址
    }
}
#endif // CHECK_WINSIZE_H