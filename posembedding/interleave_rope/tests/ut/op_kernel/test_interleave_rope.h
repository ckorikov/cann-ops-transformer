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
 * \file test_interleave_rope.h
 * \brief
 */

#ifndef _INTERLEAVE_ROPE_TILING_H_
#define _INTERLEAVE_ROPE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct InterleaveRopeTilingData {
    int64_t blockDim = 16;
    int64_t splitAxis = 4;
    int64_t batchSize = 4;
    int64_t numHead = 64;
    int64_t seqLength = 32;
    int64_t hiddenDim = 1;
    int64_t batchsPerBlock = 1;
    int64_t batchsLastBlock = 1;
    int64_t batchLoops = 1;
    int64_t batchPerLoop = 1;
    int64_t batchLastLoop = 1;
    int64_t hiddenDimCountPerBlock = 1;
    int64_t hiddenDimCountLastBlock = 1;
    int64_t hiddenDimLoopsPerBlock = 1;
    int64_t hiddenDimCountPerLoopPerBlock = 1;
    int64_t hiddenDimCountLastLoopPerBlock = 1;
    int64_t hiddenDimLoopsLastBlock = 1;
    int64_t hiddenDimCountPerLoopLastBlock = 1;
    int64_t hiddenDimCountLastLoopLastBlock = 1;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                                   \
    InterleaveRopeTilingData tilingData;                                                             \
    INIT_TILING_DATA(InterleaveRopeTilingData, tilingDataPointer, tilingPointer);                    \
    (tilingData).blockDim = tilingDataPointer->blockDim;                                             \
    (tilingData).splitAxis = tilingDataPointer->splitAxis;                                           \
    (tilingData).batchSize = tilingDataPointer->batchSize;                                           \
    (tilingData).numHead = tilingDataPointer->numHead;                                               \
    (tilingData).seqLength = tilingDataPointer->seqLength;                                           \
    (tilingData).hiddenDim = tilingDataPointer->hiddenDim;                                           \
    (tilingData).batchsPerBlock = tilingDataPointer->batchsPerBlock;                                 \
    (tilingData).batchsLastBlock = tilingDataPointer->batchsLastBlock;                               \
    (tilingData).batchLoops = tilingDataPointer->batchLoops;                                         \
    (tilingData).batchPerLoop = tilingDataPointer->batchPerLoop;                                     \
    (tilingData).batchLastLoop = tilingDataPointer->batchLastLoop;                                   \
    (tilingData).hiddenDimCountPerBlock = tilingDataPointer->hiddenDimCountPerBlock;                 \
    (tilingData).hiddenDimCountLastBlock = tilingDataPointer->hiddenDimCountLastBlock;               \
    (tilingData).hiddenDimLoopsPerBlock = tilingDataPointer->hiddenDimLoopsPerBlock;                 \
    (tilingData).hiddenDimCountPerLoopPerBlock = tilingDataPointer->hiddenDimCountPerLoopPerBlock;   \
    (tilingData).hiddenDimCountLastLoopPerBlock = tilingDataPointer->hiddenDimCountLastLoopPerBlock; \
    (tilingData).hiddenDimLoopsLastBlock = tilingDataPointer->hiddenDimLoopsLastBlock;               \
    (tilingData).hiddenDimCountPerLoopLastBlock = tilingDataPointer->hiddenDimCountPerLoopLastBlock; \
    (tilingData).hiddenDimCountLastLoopLastBlock = tilingDataPointer->hiddenDimCountLastLoopLastBlock;

#define DTYPE_X half

#endif