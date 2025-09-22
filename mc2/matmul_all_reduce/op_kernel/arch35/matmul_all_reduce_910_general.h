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
 * \file matmul_all_reduce_910_general.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_910_GENERAL_H
#define MATMUL_ALL_REDUCE_910_GENERAL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../common.h"
#include "matmul_all_reduce_base.h"
#include "../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
class MatmulAllReduce910General : public MatmulAllReduceBase<XType, YType, CoreType>
{
public:
    __aicore__ inline MatmulAllReduce910General(
        MC2GmAddrs* addrs, ArnGmAddrs* arnAddrs, MC2TilingHeader* tilingData, TPipe* tPipe)
        : MatmulAllReduceBase<XType, YType, CoreType>(addrs, nullptr, arnAddrs, tilingData, tPipe)
    {
        mc2TilingData_ = (MatmulAllReduce910TilingDataA5*)tilingData;
        this->tileInfo_.mmTiling = &mc2TilingData_->tilematmulTiling.matmulTiling;
        this->tailInfo_.mmTiling = &mc2TilingData_->tailmatmulTiling.matmulTiling;
    }

    __aicore__ inline void Process()
    {
#if (ORIG_DTYPE_X1 == DT_BF16)
        this->PreProcForBiasOnVector();
#endif

        InnerProcess(false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            InnerProcess(true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }

        this->HcclFinalize();
    }

protected:
    __aicore__ inline void InnerProcess(bool tailFlag, uint32_t turnCnt, const MC2TileInfo& tileInfo)
    {
        const MatmulTilingData* tiling =
            (tailFlag ? &mc2TilingData_->tailmatmulTiling : &mc2TilingData_->tilematmulTiling);

        MmType mmOp;
        for (uint32_t i = 0U; i < turnCnt; ++i) {
            if (block_idx < tiling->matmulTiling.usedCoreNum) {
#if defined(__DAV_C310__)
                this->tPipe_->Reset();
                mmOp.Init(
                    this->addrs_->aGM, this->addrs_->bGM, this->addrs_->cGM, this->addrs_->biasGM, nullptr,
                    this->addrs_->workspaceGM, tiling, this->tPipe_);
#else
                if (this->addFlag_ || i == 0U) {
                    this->tPipe_->Reset();
                    mmOp.Init(
                        this->addrs_->aGM, this->addrs_->bGM, this->addrs_->cGM, this->addrs_->biasGM, nullptr,
                        this->addrs_->workspaceGM, tiling, this->tPipe_);
                } else {
                    mmOp.UpdateGlobalTensor(
                        this->addrs_->aGM, this->addrs_->bGM, this->addrs_->cGM, this->addrs_->biasGM, nullptr,
                        this->addrs_->workspaceGM);
                }
#endif
                mmOp.Process();
            }
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset);
        }
    }

private:
    MatmulAllReduce910TilingDataA5* mc2TilingData_;
};

#define INVOKE_MC2_910_OP_IMPL_HELPER(opTemplateClass, bTransFlag, coreType)                                     \
    do {                                                                                                         \
        using AType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, false>;                       \
        using BType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, bTransFlag>;                  \
        using CType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>;                               \
        using BiasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS_FOR_MC2>;                 \
        using OpType =                                                                                           \
            opTemplateClass<AType, BType, CType, BiasType, MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD>; \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM};                                     \
        MatmulAllReduce910General<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType> op(                             \
            &addrs, nullptr, (MC2TilingHeader*)&tilingData, &tPipe);                                             \
        op.Init();                                                                                               \
        op.Process();                                                                                            \
    } while (0)

#define INVOKE_MC2_910_OP_IMPL(opTemplateClass, coreType)                                  \
    do {                                                                                   \
        GET_TILING_DATA_WITH_STRUCT(MatmulAllReduce910TilingDataA5, tilingData, tilingGM); \
        if (tilingData.tilematmulTiling.matmulRunInfo.transB != 0U) {                      \
            INVOKE_MC2_910_OP_IMPL_HELPER(opTemplateClass, true, coreType);                \
        } else {                                                                           \
            INVOKE_MC2_910_OP_IMPL_HELPER(opTemplateClass, false, coreType);               \
        }                                                                                  \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_910_GENERAL_H