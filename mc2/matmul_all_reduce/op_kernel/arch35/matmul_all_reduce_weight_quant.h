/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file matmul_all_reduce_weight_quant.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_WEIGHT_QUANT_H
#define MATMUL_ALL_REDUCE_WEIGHT_QUANT_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../common.h"
#include "../3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_constant.h"
#include "../3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base.h"
#include "matmul_all_reduce_base.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
using WeightQuantBatchMatmulV2::QuantType;
template <typename XType, typename WType, typename YType, class MmType>
class MatmulAllReduceWeightQuantRegBase : public MatmulAllReduceBase<XType, YType, Mc2CoreType::ON_CUBE_AND_VECTOR>
{
public:
    __aicore__ inline MatmulAllReduceWeightQuantRegBase(
        MC2GmAddrs* addrs, QuantGmAddrs* quantAddrs, ArnGmAddrs* arnAddrs, MC2TilingHeader* tilingData, TPipe* tPipe)
        : MatmulAllReduceBase<XType, YType, Mc2CoreType::ON_CUBE_AND_VECTOR>(
              addrs, quantAddrs, arnAddrs, tilingData, tPipe)
    {
        mc2TilingData_ = (WeightQuantMatmulAllReduceA5TilingData*)tilingData;
        this->tileInfo_.mmTiling = &mc2TilingData_->tileRegBaseMmTiling.matmulTiling;
        this->tailInfo_.mmTiling = &mc2TilingData_->tailRegBaseMmTiling.matmulTiling;
    }
    __aicore__ inline void Process()
    {
        InnerProcess(false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            InnerProcess(true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }
        this->HcclFinalize();
    }

protected:
    __aicore__ inline void InnerProcess(const bool tailFlag, const uint32_t turnCnt, const MC2TileInfo& tileInfo)
    {
        const WeightQuantBatchMatmulV2RegBaseTilingData* tiling =
            (tailFlag) ? &mc2TilingData_->tailRegBaseMmTiling : &mc2TilingData_->tileRegBaseMmTiling;
        for (uint32_t idx = 0; idx < turnCnt; ++idx) {
            MmType mmOp;
            this->tPipe_->Reset();
            mmOp.Init(
                this->addrs_->aGM, this->addrs_->bGM, this->quantAddrs_->antiquantScaleGM,
                this->quantAddrs_->antiquantOffsetGM, nullptr, nullptr, this->addrs_->biasGM, this->addrs_->cGM,
                this->addrs_->workspaceGM, tiling, this->tPipe_);
            mmOp.Process();
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset);
        }
    }

private:
    WeightQuantMatmulAllReduceA5TilingData* mc2TilingData_;
};

#define INVOKE_MC2_WEIGHT_QUANT_KERNEL(bTransFlag, quantType, offsetFlag, weightNz)                       \
    do {                                                                                                  \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantMatmulAllReduceA5TilingData, tilingData, tilingGM);        \
        using OpType = WeightQuantBatchMatmulV2::Arch35::WeightQuantBatchMatmulV2RegBaseKernel<           \
            DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, false, bTransFlag, offsetFlag, quantType, weightNz>; \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM};                              \
        QuantGmAddrs quantAddrs = {antiquantScaleGM, antiquantOffsetGM, nullptr, nullptr};                \
        MatmulAllReduceWeightQuantRegBase<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType> op(                        \
            &addrs, &quantAddrs, nullptr, (MC2TilingHeader*)&tilingData, &tPipe);                         \
        op.Init();                                                                                        \
        op.Process();                                                                                     \
    } while (0)

} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_WEIGHT_QUANT_H