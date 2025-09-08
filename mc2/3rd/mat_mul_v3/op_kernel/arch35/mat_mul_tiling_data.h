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
 * \file mat_mul_tiling_data.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_TILING_DATA_H__
#define __OP_KERNEL_MATMUL_TILING_DATA_H__

#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

constexpr uint64_t TILINGDATA_OFFSET = 512;
constexpr uint64_t TILINGDATA_SPLIT_NUM = 2;

#pragma pack(push, 8)
struct MatMulV3TilingData {
    TCubeTiling tCubeTiling;
    uint32_t mTailCnt = 0;
    uint32_t nTailCnt = 0;
    uint32_t kTailCnt = 0;
    uint32_t mBaseTailCnt = 0;
    uint32_t nBaseTailCnt = 0;
    uint32_t isHf32 = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct MatMulV3TilingDataCopy {
    MatMulV3TilingData matMulTilingData;
    uint8_t reserved[TILINGDATA_OFFSET] = {};  // 申请一个空的512B大小的空间，用于tiling分块
};
#pragma pack(pop)

#pragma pack(push, 8)
struct BatchMatMulV3TilingData {
    MatMulV3TilingData matMulTilingData;
    uint32_t aBatchDimAll = 1;
    uint32_t bBatchDimAll = 1;
    uint32_t cBatchDimAll = 1;
    uint32_t biasBatchDimAll = 1;
    uint32_t aBatchDim0 = 1;
    uint32_t bBatchDim0 = 1;
    uint32_t cBatchDim0 = 1;
    uint32_t aBatchDim1 = 1;
    uint32_t bBatchDim1 = 1;
    uint32_t cBatchDim1 = 1;
    uint32_t aBatchDim2 = 1;
    uint32_t bBatchDim2 = 1;
    uint32_t cBatchDim2 = 1;
    uint32_t aBatchDim3 = 1;
    uint32_t bBatchDim3 = 1;
    uint32_t cBatchDim3 = 1;
    uint32_t iterBatch = 1;
    uint32_t batchOutNum = 1;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct MatMulV3BasicTilingData {
    uint32_t usedCoreNum = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t mL1 = 0;
    uint32_t nL1 = 0;
    uint32_t kL1 = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t mTailCnt = 0;
    uint32_t nTailCnt = 0;
    uint32_t isHf32 = 0;
    uint32_t l1BufferNum = 0;
    uint32_t l0cDB = 1; // 默认不开db为1
};
#pragma pack(pop)

#pragma pack(push, 8)
struct BatchMatMulV3BasicTilingData {
    MatMulV3BasicTilingData matMulTilingData;
    uint32_t batchDimAll = 1;
    uint32_t reserved = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct BatchMatMulV3IterBatchBasicTilingData {
    uint32_t m = 1;
    uint32_t n = 1;
    uint32_t k = 1;
    uint32_t b = 1;
    uint32_t iterBatchL1 = 1;
    uint32_t iterBatchL0 = 1;
    uint32_t isHf32 = 0;
};
#pragma pack(pop)
#endif // __OP_KERNEL_MATMUL_TILING_DATA_H__
