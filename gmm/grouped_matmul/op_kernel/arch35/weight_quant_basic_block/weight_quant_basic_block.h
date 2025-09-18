/**
 * Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_basic_block.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_H

#include "basic_block_config.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "tool.h"
#include "weight_quant_basic_block_base.h"
#include "weight_quant_cube_compute.h"
#include "weight_quant_vec_compute.h"

using AscendC::GetSubBlockIdx;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;

namespace WeightQuantBatchMatmulV2::Arch35 {

#define GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM                                                                         \
    template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, typename biasType, \
              typename yType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>

#define GMM_WQ_BASIC_BLOCK_CLASS \
    WeightQuantMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, biasType, yType, wqmmConfig, vecConfig>

GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
class WeightQuantMatmulBasicBlock : public WeightQuantMatmulBasicBlockBaseClass {
public:
    __aicore__ inline WeightQuantMatmulBasicBlock(){};
    __aicore__ inline void Init(bool hasBias, uint64_t antiQuantGroupSize, uint64_t aPrefetchSize,
                                const TCubeTiling *__restrict matmulTiling, TPipe *tPipe);
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ wType *weight,
                                            __gm__ antiQuantScaleType *antiquantScale, __gm__ xType *antiquantOffset,
                                            __gm__ scaleType *scale, __gm__ float *perTokenScale, __gm__ biasType *bias,
                                            __gm__ yType *y, const bool hasBias, const bool weightL2Cacheable);
    __aicore__ inline void ComputeBasicBlock(const BasicBlockOffsetParam &offsetParam,
                                             const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit);
    __aicore__ inline void End(const BasicBlockOffsetParam &offsetParam);

protected:
    __aicore__ inline void SetAivToAic();
    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();
    __aicore__ inline void WaitAicToAiv();
    __aicore__ inline void ComputeBasicBlockAivNdNkNzKn(const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void ComputeBasicBlockAivNdKn(const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void ComputeBasicBlockAic(const BasicBlockOffsetParam &offsetParam);

    BasicBlockLibVectorAntiQuantCompute<xType, wType, antiQuantScaleType, yType, wqmmConfig, vecConfig> vectorCompute_;
    using MMImpl = MatmulImpl<MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, xType, wqmmConfig.aTrans>,
                              MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, xType, wqmmConfig.bTrans>,
                              MatmulType<TPosition::GM, CubeFormat::ND, yType>,
                              MatmulType<TPosition::TSCM, CubeFormat::ND, biasType>, CFG_MDL>;
    WeightQuantBatchMatmulV2CubeCompute<xType, biasType, yType, wqmmConfig, MMImpl> cubeCompute_;

    uint64_t cvLoopIdx_ = 0;

    LocalTensor<xType> weightF16L1_;
    uint64_t weightF16L1DbOffset_;
};

GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::Init(bool hasBias, uint64_t antiQuantGroupSize, uint64_t aPrefetchSize,
                                                      const TCubeTiling *__restrict matmulTiling, TPipe *tPipe)
{
    TBuf<TPosition::TSCM> l1Tbuf;
    uint64_t weightL1Space = matmulTiling->baseN * matmulTiling->stepKb * matmulTiling->baseK;  // weight单块大小
    if constexpr (IsSameType<yType, int8_t>::value) {
        tPipe->InitBuffer(l1Tbuf, 504 * 1024);  // 除去quantScale, 共使用504KB
        weightF16L1DbOffset_ = 504 * GetKBUnit<half>() - weightL1Space;
    } else {
        tPipe->InitBuffer(l1Tbuf, 512 * 1024);
        weightF16L1DbOffset_ = 512 * GetKBUnit<half>() - weightL1Space;
    }
    weightF16L1_ = l1Tbuf.Get<xType>();
    if ASCEND_IS_AIC {
        cubeCompute_.Init(l1Tbuf, weightL1Space, aPrefetchSize, matmulTiling, tPipe);
    } else {
        vectorCompute_.Init(tPipe);
    }
}

GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::UpdateGlobalAddr(__gm__ xType *x, __gm__ wType *weight,
                                                                  __gm__ antiQuantScaleType *antiquantScale,
                                                                  __gm__ xType *antiquantOffset,
                                                                  __gm__ scaleType *scale, __gm__ float *perTokenScale,
                                                                  __gm__ biasType *bias, __gm__ yType *y,
                                                                  const bool hasBias, const bool weightL2Cacheable)
{
    if ASCEND_IS_AIC {
        cubeCompute_.UpdateGlobalAddr(x, y, bias, scale, hasBias);
    } else {
        vectorCompute_.UpdateGlobalAddr(weight, antiquantScale, antiquantOffset, nullptr, nullptr, nullptr,
                                        weightL2Cacheable);
    }
}

/*
 * 该函数作用为根据转置属性，L1上shape大小以及vecconfig确定vec核的实际搬运量
 * ND TransB = True  两个vec核的mte2搬运量为 (curVecCoreMte2RealN, curVecCoreMte2RealK)
 * NZ TransB = False 两个vec核的mte2搬运量为 (CeilDiv(curVecCoreMte2RealN,16), CeilDiv(curVecCoreMte2RealK,16), 16 ,16)
 */
GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::ComputeBasicBlockAivNdNkNzKn(const BasicBlockOffsetParam &offsetParam)
{
    UbConsumeConfig ubConsumeConfig;
    L1ConsumeConfig l1ConsumeConfig;
    ubConsumeConfig.nWeightLowBitUbOffset = 0;
    l1ConsumeConfig.l1RealExternalLen = offsetParam.nL1Size;

#if defined(__DAV_310R6__)
    uint64_t curVecCoreMte2RealN = offsetParam.nL1Size;
#else
    // 初值为一半的nl1
    uint64_t curVecCoreMte2RealN = offsetParam.nL1Size >> 1;
    if constexpr (wqmmConfig.weightFormat == CubeFormat::NZ) {
        curVecCoreMte2RealN = offsetParam.nL1Size > BLOCK_CUBE
                                  ? CeilAlign(curVecCoreMte2RealN, static_cast<uint64_t>(BLOCK_CUBE))
                                  : offsetParam.nL1Size;
    }
#endif
    // 实际值需要根据vec核来确定
    ubConsumeConfig.l1RequireVfComputeRealN =
        GetSubBlockIdx() == 0 ? curVecCoreMte2RealN : offsetParam.nL1Size - curVecCoreMte2RealN;
    l1ConsumeConfig.l1SplitTwoVecExternalOffset = GetSubBlockIdx() * curVecCoreMte2RealN;

    for (uint64_t kMte2Offset = 0; kMte2Offset < offsetParam.kSize; kMte2Offset += vecConfig.ubMte2InnerSize) {
        uint64_t mte2RealK = (kMte2Offset + vecConfig.ubMte2InnerSize) > offsetParam.kSize
                                 ? offsetParam.kSize - kMte2Offset
                                 : vecConfig.ubMte2InnerSize;  // vec总共需要搬运的K方向的实际量（考虑尾块）
        vectorCompute_.WaitVToMTE2();
        vectorCompute_.CopyGmToUb(ubConsumeConfig.l1RequireVfComputeRealN, mte2RealK,
                                  offsetParam.nOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset, kMte2Offset,
                                  offsetParam);

        // 当前方案下，不会出现N方向计算量小于载入量的情况，所以没有N的循环
        for (ubConsumeConfig.kWeightLowBitUbOffset = 0; ubConsumeConfig.kWeightLowBitUbOffset < mte2RealK;
             ubConsumeConfig.kWeightLowBitUbOffset += offsetParam.kbL1Size, cvLoopIdx_++) {
            ubConsumeConfig.l1RequireVfComputeRealK =
                (ubConsumeConfig.kWeightLowBitUbOffset + offsetParam.kbL1Size) >= mte2RealK
                    ? mte2RealK - ubConsumeConfig.kWeightLowBitUbOffset
                    : offsetParam.kbL1Size;
            if (cvLoopIdx_ > 1) {
                WaitAicToAiv();
            }
            vectorCompute_.WeightAntiQuantCompute(
                ubConsumeConfig, weightF16L1_[(cvLoopIdx_ & 1) * weightF16L1DbOffset_], l1ConsumeConfig);
            SetAivToAic();
        }
        vectorCompute_.SetVToMTE2();
    }
}

/*
 * 该函数作用为根据转置属性，L1上shape大小以及vecconfig确定vec核的实际搬运量
 * ND TransB = False 两个vec核的mte2搬运量为 (curVecCoreMte2RealK, curVecCoreMte2RealN)
 */
GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::ComputeBasicBlockAivNdKn(const BasicBlockOffsetParam &offsetParam)
{
// nd-kn场景下，搬运消费比为1:1
#if defined(__DAV_310R6__)
    // 此时c:v为1：1, cube所需数据由一个v提供
    uint64_t kMte2BaseSize = offsetParam.kbL1Size;
#else
    // 此时c:v为1：2, cube所需数据由两个v提供
    uint64_t kMte2BaseSize = offsetParam.kbL1Size >> 1;
#endif

    UbConsumeConfig ubConsumeConfig;
    L1ConsumeConfig l1ConsumeConfig;
    ubConsumeConfig.l1RequireVfComputeRealN = offsetParam.nL1Size;
    ubConsumeConfig.kWeightLowBitUbOffset = 0;
    ubConsumeConfig.nWeightLowBitUbOffset = 0;
    l1ConsumeConfig.l1SplitTwoVecExternalOffset = GetSubBlockIdx() * kMte2BaseSize;

    for (uint64_t kMte2Offset = 0; kMte2Offset < offsetParam.kSize; kMte2Offset += offsetParam.kbL1Size, cvLoopIdx_++) {
        l1ConsumeConfig.l1RealExternalLen = (kMte2Offset + offsetParam.kbL1Size) > offsetParam.kSize
                                                ? offsetParam.kSize - kMte2Offset
                                                : offsetParam.kbL1Size;
        /*
         * 场景1：当前core为v0，mte2实际搬运的k值为mte2方向搬运标准值(kMte2BaseSize)和l1上k实际值的最小值
         * 场景2: 当前core为v1, 且l1实际的k值比core v1搬运值大，则mte2实际搬运的k值为两者之差
         * 场景3：当前core为v1, 且l1实际的k值比core v1搬运值小，则mte2无需搬运,设置为0即可
         */
        uint64_t mte2RealK = GetSubBlockIdx() == 0 ? min(kMte2BaseSize, l1ConsumeConfig.l1RealExternalLen)
                             : l1ConsumeConfig.l1RealExternalLen > kMte2BaseSize
                                 ? l1ConsumeConfig.l1RealExternalLen - kMte2BaseSize
                                 : 0;
        vectorCompute_.WaitVToMTE2();
        vectorCompute_.CopyGmToUb(offsetParam.nL1Size, mte2RealK, offsetParam.nOffset,
                                  kMte2Offset + GetSubBlockIdx() * kMte2BaseSize, offsetParam);

        if (cvLoopIdx_ > 1) {
            WaitAicToAiv();
        }
        ubConsumeConfig.l1RequireVfComputeRealK = mte2RealK;
        vectorCompute_.WeightAntiQuantCompute(ubConsumeConfig, weightF16L1_[(cvLoopIdx_ & 1) * weightF16L1DbOffset_],
                                              l1ConsumeConfig);
        SetAivToAic();
        vectorCompute_.SetVToMTE2();
    }
}

GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::ComputeBasicBlockAic(const BasicBlockOffsetParam &offsetParam)
{
    // 当前方案下，不会出现N方向计算量小于载入量的情况，所以没有N的循环
    for (uint64_t kbL1Offset = 0; kbL1Offset < offsetParam.kSize; kbL1Offset += offsetParam.kbL1Size, cvLoopIdx_++) {
        uint64_t kbL1RealSize = (kbL1Offset + offsetParam.kbL1Size) >= offsetParam.kSize
                                    ? offsetParam.kSize - kbL1Offset
                                    : offsetParam.kbL1Size;
        cubeCompute_.WaitMTE1ToMTE2(cvLoopIdx_);
        cubeCompute_.CopyAAndBiasGmToL1(offsetParam, kbL1Offset, kbL1RealSize, offsetParam.nL1Size, cvLoopIdx_);
        WaitAivToAic();
        cubeCompute_.LaunchMatmul(weightF16L1_[(cvLoopIdx_ & 1) * weightF16L1DbOffset_], kbL1Offset, kbL1RealSize,
                                  offsetParam, cvLoopIdx_);  // mte1 mmad fixp流水
        cubeCompute_.SetMTE1ToMTE2(cvLoopIdx_);
        SetAicToAiv();
    }
    cubeCompute_.GetTensorC(offsetParam);
    cubeCompute_.ClearAFullLoadFlag();  // 清除A全载时之前循环的set同步标记
}

GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::ComputeBasicBlock(const BasicBlockOffsetParam &offsetParam,
                                                                   const BasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIV {
        if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ && !wqmmConfig.bTrans) {
            ComputeBasicBlockAivNdKn(offsetParam);
        } else {
            ComputeBasicBlockAivNdNkNzKn(offsetParam);
        }
    } else {
        ComputeBasicBlockAic(offsetParam);
    }
}

GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit)
{
    cubeCompute_.PrefetchA(aPrefetchSize, xSizeLimit);
}

GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::End(const BasicBlockOffsetParam &offsetParam)
{
    if ASCEND_IS_AIC {
        cubeCompute_.EndSync(cvLoopIdx_);
    } else {
        if (cvLoopIdx_ > 0) {
            WaitAicToAiv();
        }
        if (cvLoopIdx_ > 1) {
            WaitAicToAiv();
        }
        vectorCompute_.End();
    }
}

GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::SetAivToAic()
{
#ifndef __CCE_KT_TEST__
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
#endif
}

GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::WaitAivToAic()
{
#ifndef __CCE_KT_TEST__
#ifndef __DAV_310R6__
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
#endif
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
#endif
}

GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::SetAicToAiv()
{
#ifndef __CCE_KT_TEST__
#if !defined(__DAV_310R6__)
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
#endif
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
#endif
}

GMM_WQ_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_BASIC_BLOCK_CLASS::WaitAicToAiv()
{
#ifndef __CCE_KT_TEST__
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
#endif
}
}  // namespace WeightQuantBatchMatmulV2::Arch35

#endif  // GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_H
