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
 * \file grouped_matmul.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_SAME_WEIGHT_H
#define ASCENDC_GROUPED_MATMUL_SAME_WEIGHT_H

#include "grouped_matmul_utils.h"
#include "grouped_matmul.h"

namespace GROUPED_MATMUL {

/** @brief GroupMatmul operator Class
*/
template <typename ComputeType>
class GMMGroupProcessSameWeight : public GMMProcess<ComputeType> {
 public:
    /** @brief constructor */
    __aicore__ inline GMMGroupProcessSameWeight(ComputeType& computeOp_) : GMMProcess<ComputeType>(computeOp_) {}
    __aicore__ inline void Process() {
        MNConfig mnConfig;
        if (this->gmmBaseParams->groupType != -1) {  // -1: no split
            if (unlikely(this->groupListPtr == nullptr)) {
                return;
            }
            this->preOffset = 0;
        }
        AscendC::WaitPreTaskEnd();
        for (uint32_t groupIdx = 0, count = 0; groupIdx < this->groupNum; ++groupIdx) {
            this->UpdateMnConfig(mnConfig);
            int32_t splitValue = GetSplitValueFromGroupList(groupIdx, this->preOffset, this->gmmBaseParams, this->groupListGm);
            if (groupIdx == 0) {
                this->SetMNConfig(splitValue, groupIdx, mnConfig);
            } else {
                mnConfig.m = splitValue;
            }
            if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
                continue;
            }
            mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
            mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);

            uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
            uint32_t curBlock = this->coreIdx >= count ? this->coreIdx : this->coreIdx + this->gmmBaseParams->coreNum;
            uint32_t thresholdM_dimN = thresholdBlockNum * mnConfig.blockDimN;

            while (curBlock < curCount) {
                MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
                this->computeOp.MMCompute(groupIdx, mnConfig, this->coreIdx);
                this->computeOp.VectorCompute(mnConfig);
                curBlock += this->gmmBaseParams->coreNum;
            }
            count = curCount % this->gmmBaseParams->coreNum;
        }
        this->computeOp.PostCompute();
        AscendC::SetNextTaskStart();
    }

};

}  // namespace GROUPED_MATMUL

#endif  // ASCENDC_GROUPED_MATMUL_H
