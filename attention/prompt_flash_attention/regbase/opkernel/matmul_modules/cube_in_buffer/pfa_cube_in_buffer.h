/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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
 * \file pfa_cube_in_buffer.h
 * \brief
 */
#ifndef PFA_CUBE_IN_BUFFER_H
#define PFA_CUBE_IN_BUFFER_H

#include "../pfa_policy_data.h"

constexpr int32_t ALIGN_16_MASK = 0xFFFFFFF0;
constexpr int32_t ALIGN_16_OFFSET = 15; // 向上16对齐

__aicore__ inline int32_t Align16Func(int32_t data) {
    return (data + ALIGN_16_OFFSET) & ALIGN_16_MASK;
}

namespace AscendC {
namespace Impl {
namespace Detail {
template<typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class PFACubeInBuffer {
    using SrcT = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline PFACubeInBuffer() {}
    __aicore__ inline ~PFACubeInBuffer() {}

    __aicore__ inline void Init(const MatmulTiling<MM_CFG>& cubeTiling, int32_t baseSize, int32_t cacheSize,
                                int32_t reduceAxisCnt) {}

    __aicore__ inline void Destroy() {
        tscmGlobalPFA->localScm[tscmIndex_].FreeTensor(cacheHead_);
    }

    __aicore__ inline LocalTensor<SrcT> AllocTensor(int32_t iterIndex) {
        cacheHead_ = tscmGlobalPFA->localScm[iterIndex].template AllocTensor<SrcT>();
        tscmIndex_ = iterIndex;
        return cacheHead_;
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<SrcT>& tensor = NULL_TENSOR<SrcT>) {}

    __aicore__ inline void Reset() {
        tscmGlobalPFA->localScm[tscmIndex_].FreeTensor(cacheHead_);
    }

    __aicore__ inline LocalTensor<SrcT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1) {
        return cacheHead_;
    }

    __aicore__ inline void EnQue(LocalTensor<SrcT>& tensor) {
        (void)tscmGlobalPFA->localScm[tscmIndex_].EnQue(tensor);
    }

    __aicore__ inline void DeQue() {
        (void)tscmGlobalPFA->localScm[tscmIndex_].DeQue();
    }

    __aicore__ inline void SetOrgAddr(__gm__ SrcT* gmAddr) {}

    __aicore__ inline uint64_t GetBufferHeadAddr()
    {
        return GetTQueHeadAddr(tscmGlobalPFA->localScm[tscmIndex_]);
    }

private:
    LocalTensor<SrcT> cacheHead_;
    int32_t tscmIndex_;
};

}
}
}
#endif // PFA_CUBE_IN_BUFFER_H