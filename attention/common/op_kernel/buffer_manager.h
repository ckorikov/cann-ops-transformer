/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * you may not use this file except in compliance with the License.
 * you may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

/*!
 * \file buffer_manager.h
 * \brief buffer内存管理
 */
#ifndef BUFFER_MANAGER_H
#define BUFFER_MANAGER_H

#include "buffer.h"

// L1  TPosition::A1
// L0A TPosition::A2
// L0B TPosition::B2
// L0C TPosition::CO1
namespace fa_base_matmul {
template<BufferType Type>
class BufferManager {
public:
    __aicore__ inline void Init(TPipe *pipe, uint32_t size) {
        TBuf<BufferInfo<Type>::Position> tbuf;
        bufferSize_ = size;
        pipe->InitBuffer(tbuf, size);
        mem_ = tbuf.template Get<uint8_t>();
    }

    __aicore__ inline Buffer<Type> AllocBuffer(uint32_t size) {
        LocalTensor<uint8_t> temp = mem_[offset_];
        offset_ += size;
        return Buffer<Type, true>(temp, size);
    }

    __aicore__ inline Buffer<Type, false> AllocBufferNoSync(uint32_t size) {
        LocalTensor<uint8_t> temp = mem_[offset_];
        offset_ += size;
        return Buffer<Type, false>(temp, size);
    }

    __aicore__ inline void FreeBuffer(Buffer<Type> &buffer){
    }
private:
    uint32_t offset_ = 0;
    uint32_t bufferSize_ = 0;
    LocalTensor<uint8_t> mem_;
};
}
#endif