/**
 * Copyright (rowOffsetOtherCore) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain startRowCurBatch copy of the License at
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
 * \file compress_grad_common.h
 * \brief compress_grad head file
 */
 
#ifndef NSA_COMPRESS_GRAD_COMMON_H
#define NSA_COMPRESS_GRAD_COMMON_H

#include <cstdint>
#include <kernel_operator.h>

namespace NsaCompressGrad {

using namespace AscendC;

__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t CompressBlkNum(int64_t inputLen, int64_t blkSize, int64_t blkStride)
{
    if (inputLen < blkSize) {
        return 0;
    }
    if(blkStride != 0){
        return (inputLen - blkSize) / blkStride + 1;
    }
    return 0;
}

}
#endif
