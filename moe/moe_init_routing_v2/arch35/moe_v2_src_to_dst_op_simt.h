/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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

/* !
 * \file moe_src_to_dst_op_simt.h
 * \brief
 */
#ifndef MOE_V2_SRC_TO_DST_SIMT_H
#define MOE_V2_SRC_TO_DST_SIMT_H

#include "moe_v2_common.h"

namespace MoeInitRoutingV2 {
using namespace AscendC;

class MoeV2SrcToDstOpSimt {
public:
    __aicore__ inline MoeV2SrcToDstOpSimt(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expandDstToSrcRow, const TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SyncAll();

private:
    __gm__ int32_t *expandDstToSrcRowGm_;
    __gm__ int32_t *expandedRowIdxGm_;
    const MoeV2GatherOutComputeTilingData *srcToDstTilingData_;

    int64_t coreNum_;
    int64_t blockIdx_;
    int64_t totalLength_;
    int64_t perCoreRows_;
    int64_t coreRows_;
    int64_t startIndex_;
    int64_t threadNum_;
};

__aicore__ inline void MoeV2SrcToDstOpSimt::SyncAll()
{
    if (coreNum_ == 1) {
        return;
    }
    AscendC::SyncAll();
}

template <typename TilingData>
__aicore__ inline void MoeV2SrcToDstOpSimt::Init(GM_ADDR expandedRowIdx, GM_ADDR expandDstToSrcRow,
                                                 const TilingData *tilingData)
{
    this->blockIdx_ = GetBlockIdx();
    this->coreNum_ = tilingData->coreNum;
    this->totalLength_ = tilingData->n * tilingData->k;
    this->srcToDstTilingData_ = &(tilingData->srcToDstComputeParamsOp);
    this->perCoreRows_ = this->srcToDstTilingData_->perCoreRows;
    if (this->blockIdx_ == this->srcToDstTilingData_->needCoreNum - 1) {
        this->coreRows_ = this->srcToDstTilingData_->lastCoreRows;
    } else {
        this->coreRows_ = this->srcToDstTilingData_->perCoreRows;
    }
    startIndex_ = this->blockIdx_ * this->perCoreRows_;
    this->threadNum_ = THREAD_NUM < this->coreRows_ ? THREAD_NUM : this->coreRows_;

    expandedRowIdxGm_ = (__gm__ int32_t *)expandedRowIdx;
    expandDstToSrcRowGm_ = (__gm__ int32_t *)expandDstToSrcRow + Align(this->totalLength_, sizeof(int32_t));
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ComputeSimt(int64_t coreRows, int64_t startIndex,
                                                                        __gm__ int32_t *expandDstToSrcRowGm,
                                                                        __gm__ int32_t *expandedRowIdxGm)
{
    for (int32_t index = static_cast<int32_t>(Simt::GetThreadIdx()); index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(Simt::GetThreadNum())) {
        int64_t srcIndex = index + startIndex;
        int64_t dstIndex = expandDstToSrcRowGm[srcIndex];
        expandedRowIdxGm[dstIndex] = srcIndex;
    }
}

__aicore__ inline void MoeV2SrcToDstOpSimt::Process()
{
    if (this->blockIdx_ < this->srcToDstTilingData_->needCoreNum) {
        Simt::VF_CALL<ComputeSimt>(Simt::Dim3{static_cast<uint32_t>(this->threadNum_), 1, 1}, this->coreRows_,
                                   this->startIndex_, expandDstToSrcRowGm_, expandedRowIdxGm_);
    }
    this->SyncAll();
}
} // namespace MoeInitRoutingV2
#endif // MOE_SRC_TO_DST_SIMT_H