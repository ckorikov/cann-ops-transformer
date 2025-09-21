/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file ifa_cube_in_buffer.h
 * \brief
 */
#ifndef IFA_CUBE_IN_BUFFER_H
#define IFA_CUBE_IN_BUFFER_H

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class IFACubeInBuffer {
  using SRC_T = typename INPUT_TYPE::TRANS_T;

 public:
  __aicore__ inline IFACubeInBuffer() {
  }
  __aicore__ inline ~IFACubeInBuffer() {
  }

  __aicore__ inline void Init() {
    auto tPipePtr = GetTPipePtr();
    tPipePtr->InitBuffer(queryScm[0], 1, BUFFER_SIZE_BYTE_16K);
    tPipePtr->InitBuffer(queryScm[1], 1, BUFFER_SIZE_BYTE_16K);
    tPipePtr->InitBuffer(queryScm[2], 1, BUFFER_SIZE_BYTE_16K);
    tPipePtr->InitBuffer(queryScm[3], 1, BUFFER_SIZE_BYTE_16K);
  }

  __aicore__ inline void Destroy() {
  }

  __aicore__ inline LocalTensor<SRC_T> AllocTensor(int32_t idx) {
    lastTensor = queryScm[idx].template AllocTensor<SRC_T>();
    tscmIdx = idx;
    return lastTensor;
  }

  __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<SRC_T>& tensor = NULL_TENSOR<SRC_T>) {
    queryScm[tscmIdx].FreeTensor(lastTensor);
  }

  __aicore__ inline void Reset() {
  }

  __aicore__ inline bool Hit(int32_t iterIndex, int32_t bufferPos = -1) {
  }

  __aicore__ inline LocalTensor<SRC_T> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1) {
    return NULL_TENSOR<SRC_T>;
  }

  __aicore__ inline void EnQue(LocalTensor<SRC_T>& tensor) {
    queryScm[tscmIdx].EnQue(tensor);
  }

  __aicore__ inline void DeQue() {
    queryScm[tscmIdx].DeQue();
  }

  __aicore__ inline void SetOrgAddr(__gm__ SRC_T* gmAddr) {
  }

  __aicore__ inline int32_t GetIterIndex(int32_t curRow, int32_t curCol) {
  }

 private:
  TSCM<TPosition::GM, 1, 0x04> queryScm[4];
  LocalTensor<SRC_T> lastTensor;
  int32_t tscmIdx;
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif  // IFA_CUBE_IN_BUFFER_H