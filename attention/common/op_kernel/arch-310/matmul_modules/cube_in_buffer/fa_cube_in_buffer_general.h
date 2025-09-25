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
 * \file fa_cube_in_buffer_general.h
 * \brief
 */

#ifndef FA_CUBE_IN_BUFFER_GENERAL_H
#define FA_CUBE_IN_BUFFER_GENERAL_H

#include "../fa_flag_data.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
constexpr int ADD_NUM_15 = 15;
constexpr uint32_t LOWER_4_BITS_ZERO = 0xFFFFFFF0;

__aicore__ inline int Align16Func(int data) {
    return (data + ADD_NUM_15) & LOWER_4_BITS_ZERO;
}

template<typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class FACubeInBufferGeneral {
    using SrcT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline FACubeInBufferGeneral() {}
    __aicore__ inline ~FACubeInBufferGeneral() {}

    __aicore__ inline void Init(int32_t baseSize, int32_t cacheSize, int32_t reduceAxisCnt) {}

    __aicore__ inline void Destroy() {
        tscmGlobal->localQue[tscmIndex_].FreeTensor(cacheHead_);
    }

    __aicore__ inline LocalTensor<SrcT> AllocTensor(int32_t iterIndex) {
        cacheHead_ = tscmGlobal->localQue[iterIndex].template AllocTensor<SrcT>();
        tscmIndex_ = iterIndex;
        return cacheHead_;
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, LocalTensor<SrcT>& tensor = NULL_TENSOR<SrcT>) {}

    __aicore__ inline void Reset() {
        tscmGlobal->localQue[tscmIndex_].FreeTensor(cacheHead_);
    }

    __aicore__ inline LocalTensor<SrcT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1) {
        return cacheHead_;
    }

    __aicore__ inline void SetOrgAddr(__gm__ SrcT* gmAddr) {}

    __aicore__ inline void EnQue(LocalTensor<SrcT>& tensor) {
        tscmGlobal->localQue[tscmIndex_].EnQue(tensor);
    }

    __aicore__ inline void DeQue() {
        tscmGlobal->localQue[tscmIndex_].DeQue();
    }

private:
    int32_t tscmIndex_;
    LocalTensor<SrcT> cacheHead_;
};
}
}
}

#endif // FA_CUBE_IN_BUFFER_GENERAL_H
