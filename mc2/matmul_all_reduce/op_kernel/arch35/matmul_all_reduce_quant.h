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
 * \file matmul_all_reduce_quant.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_QUANT_H
#define MATMUL_ALL_REDUCE_QUANT_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../common.h"

#include "kernel_operator_intf.h"
#include "matmul_all_reduce_base.h"
#include "../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
class MatmulAllReduceQuant : public MatmulAllReduceBase<XType, YType, CoreType>
{
public:
    __aicore__ inline MatmulAllReduceQuant(
        MC2GmAddrs* addrs, QuantGmAddrs* quantAddrs, ArnGmAddrs* arnAddrs, MC2TilingHeader* tilingData, TPipe* tPipe,
        bool isMX)
        : MatmulAllReduceBase<XType, YType, CoreType>(addrs, quantAddrs, arnAddrs, tilingData, tPipe)
    {
        mc2TilingData_ = (QuantMatmulAllReduceTilingDataA5*)tilingData;
        this->tileInfo_.mmTiling = &mc2TilingData_->tilematmulTiling.matmulTiling;
        this->tailInfo_.mmTiling = &mc2TilingData_->tailmatmulTiling.matmulTiling;
        isMXScene_ = isMX;
    }

    __aicore__ inline void Process()
    {
        MmType opTile;
        InnerProcess(opTile, false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            MmType opTail;
            InnerProcess(opTail, true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }

        this->HcclFinalize();
    }

protected:
    __aicore__ inline void InnerProcess(MmType& mmOp, bool tailFlag, uint32_t turnCnt, const MC2TileInfo& tileInfo)
    {
        const QuantBatchMatmulV3TilingData* tiling =
            (tailFlag ? &mc2TilingData_->tailmatmulTiling : &mc2TilingData_->tilematmulTiling);
        uint64_t pertokenOffset = 0UL;
        uint64_t MX_GROUP_SIZE = 64UL;
        uint64_t NUM_TWO = 2;
        if (isMXScene_) {
            pertokenOffset =
                sizeof(AscendC::fp8_e8m0_t) * tiling->matmulTiling.M *
                CeilDiv(tiling->matmulTiling.Ka, MX_GROUP_SIZE) * NUM_TWO;
        } else {
            pertokenOffset = sizeof(float) * tiling->matmulTiling.M;
        }
        for (uint32_t i = 0U; i < turnCnt; ++i) {
            this->tPipe_->Reset();
            // 当前MM的UpdateGlobalAddr接口存在问题，暂时每轮计算均使用Init接口更新地址
            mmOp.Init(
                this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM, this->quantAddrs_->dequantGM,
                this->quantAddrs_->pertokenGM, this->addrs_->cGM, this->addrs_->workspaceGM, tiling, this->tPipe_);

            mmOp.Process();
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset);
            this->quantAddrs_->pertokenGM += pertokenOffset;
        }
    }

private:
    QuantMatmulAllReduceTilingDataA5* mc2TilingData_;
    bool isMXScene_ = false;
};

#define INVOKE_MC2_QUANT_910_OP_IMPL(templateClass, coreType, ...)                                           \
    do {                                                                                                     \
        GET_TILING_DATA_WITH_STRUCT(QuantMatmulAllReduceTilingDataA5, tilingData, tilingGM);                 \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM};                                 \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, nullptr, dequantGM, pertokenGM};                        \
        using OpType = templateClass<                                                                        \
            DTYPE_X1, DTYPE_X2, uint64_t, DTYPE_BIAS, DTYPE_Y, X1_FORMAT, X2_FORMAT, Y_FORMAT, __VA_ARGS__>; \
        MatmulAllReduceQuant<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType> op(                              \
            &addrs, &quantAddrs, nullptr, (MC2TilingHeader*)&tilingData, &tPipe, false);                     \
        op.Init();                                                                                           \
        op.Process();                                                                                        \
    } while (0)

#define INVOKE_MC2_QUANT_MXFP_910_OP_IMPL(templateClass, coreType, ...)                                   \
    do {                                                                                                  \
        GET_TILING_DATA_WITH_STRUCT(QuantMatmulAllReduceTilingDataA5, tilingData, tilingGM);              \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM};                              \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, nullptr, dequantGM, pertokenGM};                     \
        using OpType = templateClass<                                                                     \
            DTYPE_X1, DTYPE_X2, AscendC::fp8_e8m0_t, DTYPE_BIAS, DTYPE_Y, X1_FORMAT, X2_FORMAT, Y_FORMAT, \
            __VA_ARGS__>;                                                                                 \
        MatmulAllReduceQuant<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType> op(                           \
            &addrs, &quantAddrs, nullptr, (MC2TilingHeader*)&tilingData, &tPipe, true);                   \
        op.Init();                                                                                        \
        op.Process();                                                                                     \
    } while (0)

} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_H