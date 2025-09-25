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
 * \file fa_cube_in_buffer.h
 * \brief
 */
#ifndef FA_CUBE_IN_BUFFER_H
#define FA_CUBE_IN_BUFFER_H

#include "../fa_flag_data.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template<typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class FACubeInBuffer {
    using SrcT = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline FACubeInBuffer() {}
    __aicore__ inline ~FACubeInBuffer() {}

    __aicore__ inline void Init(const MatmulTiling<MM_CFG>& cubeTiling, int32_t baseSize, int32_t cacheSize,
        int32_t reduceAxisCnt) {}

    __aicore__ inline void Destroy() {
        tscmGlobal->localQue[tscmIndex_].FreeTensor(cacheHead_);
    }

    __aicore__ inline LocalTensor<SrcT> AllocTensor(int32_t iterIndex) {
        cacheHead_ = tscmGlobal->localQue[iterIndex].template AllocTensor<SrcT>();
        tscmIndex_ = iterIndex;
        return cacheHead_;
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<SrcT>& tensor = NULL_TENSOR<SrcT>) {}

    __aicore__ inline void Reset() {
        tscmGlobal->localQue[tscmIndex_].FreeTensor(cacheHead_);
    }

    __aicore__ inline LocalTensor<SrcT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1) {
        return cacheHead_;
    }

    __aicore__ inline void EnQue(LocalTensor<SrcT>& tensor) {}

    __aicore__ inline void DeQue() {}

    __aicore__ inline void SetOrgAddr(__gm__ SrcT* gmAddr) {}

private:
    LocalTensor<SrcT> cacheHead_;
    int32_t tscmIndex_;
};
}
}
}

#endif // FA_CUBE_IN_BUFFER_H