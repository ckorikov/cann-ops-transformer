/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MLA_PREPROCESS_H
#define MLA_PREPROCESS_H

#include <cstdint>

namespace optiling {
struct MlaPpMatmulTilingData {
    int64_t numBatch{0};
    int64_t m{0};
    int64_t k{0};
    int64_t n{0};
    int64_t m0{0};
    int64_t k0{0};
    int64_t n0{0};
    int64_t mLoop{0};
    int64_t kLoop{0};
    int64_t nLoop{0};
    int64_t coreLoop{0};
    int64_t swizzleCount{0};
    int64_t swizzleDirect{0};
    int64_t enShuffleK{0};
    int64_t blockDim{0};
    int64_t enLoadAllAmat{0};
    int64_t b0matPingPongBufferLen{0};
};

struct MlaTilingData {
    int64_t numCore{0};
    int64_t n{0};
    int64_t perTaskNum{0};
    int64_t resTaskNum{0};
    MlaPpMatmulTilingData mm1;
    MlaPpMatmulTilingData mm2;
    MlaPpMatmulTilingData mm3;
    // rms1
    int64_t rmsNumCore1{0};
    int64_t rmsNumCol1{0};
    int64_t rmsNumRow1{0};
    int64_t rmsQuantMin1{0};
    int64_t hiddtenState{0};
    // rms2
    int64_t rmsNumCore2{0};
    int64_t rmsNumCol2{0};
    int64_t rmsNumRow2{0};
    int64_t rmsQuantMin2{0};

    int64_t hiddenSizeQ{0};
    int64_t headNumQ{0};
    int64_t headDim{0};
    int64_t concatSize{0};
    int64_t rotaryCoeff{0};
    int64_t ntokens{0};
    int64_t realCore{0};
    int64_t nlCoreRun{0};
    int64_t lCoreRun{0};
    int64_t maxNPerLoopForUb{0};
    int64_t preCoreLoopTime{0};
    int64_t preCoreLoopNLast{0};
    int64_t lastCoreLoopTime{0};
    int64_t lastCoreLoopNLast{0};
    // EinSumQuant
    int64_t esqFrontCore{0};
    int64_t esqTailCore{0};
    int64_t esqFrontCoreBatch{0};
    int64_t esqTailCoreBatch{0};
    int64_t esqHeadNum{0};
    int64_t esqColNum{0};
    int64_t esqUbHeadLoop{0};
    int64_t esqHeadPerLoop{0};
    int64_t esqHeadTail{0};
    int64_t esqColLoop{0};
    int64_t esqColTail{0};
    // workspace的参数配置
    int64_t maxWorkspaceSize{0};
    int64_t epsilon{0};
};
} // namespace optiling

#endif // MLA_PREPROCESS_H