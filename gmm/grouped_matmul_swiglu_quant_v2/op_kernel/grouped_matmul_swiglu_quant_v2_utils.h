/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grrouped_matmul_swiglu_quant_spilit_fusion.h
 * \brief
 */

#ifndef GROUPED_MATMUL_DEQUANT_SWIGLU_QUANT_V2_UTILS_H
#define GROUPED_MATMUL_DEQUANT_SWIGLU_QUANT_V2_UTILS_H

namespace GroupedMatmulDequantSwigluQuant {
using namespace AscendC;


constexpr uint32_t INT8_BITS = 8;
constexpr int32_t MKN_LIST_LEN = 128;
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32;
constexpr uint32_t UB_BLOCK_DOUBLE_UNIT_SIZE = 64;
constexpr uint32_t HALF_UB_BLOCK_UNIT_SIZE = UB_BLOCK_UNIT_SIZE / 2;

constexpr MatmulConfig matmulCFGUnitFlag{false, false, true, 0, 0, 0, false, false, false, false, false, 0, 0, 0,
                                        0, 0, 0, 0, false};
constexpr MatmulConfig NZ_CFG_MDL = GetMDLConfig(false, false, 0, true, false, false, false);

template <class AT_, class BT_, class CT_, class BiasT_, const MatmulConfig& MMCFG_>
struct MMImplType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    using BiasT = BiasT_;
    using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, MMCFG_>;
};

struct MNConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint64_t wBaseOffset = 0;
    uint64_t mAxisBaseOffset = 0;
    uint64_t nAxisBaseOffset = 0;
    uint64_t xBaseOffset = 0;
    uint64_t yBaseOffset = 0;
    uint64_t wOutOffset = 0;
    uint64_t workspaceOffset = 0;
};

struct VecConfig {
    int64_t M = 0;
    int64_t usedCoreNum = 0;
    int64_t startOffset = 0;
    int64_t curOffset = 0;
    int64_t startIdx = 0;
    int64_t curIdx = 0;
    int64_t taskNum = 0;
    int64_t curGroupIdx = 0;
    int64_t outLoopNum = 0;
    int64_t tailLoopNum = 0;
    int64_t NextUpdateInterVAl = 0;
};

template <typename T>
__aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;

    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}
} // namespace GroupedMatmulDequantSwigluQuant

#endif