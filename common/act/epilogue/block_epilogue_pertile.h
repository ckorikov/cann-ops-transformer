/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_pertile.h
 * \brief
 */

#ifndef EPILOGUE_BLOCK_EPILOGUE_PERTILE_H
#define EPILOGUE_BLOCK_EPILOGUE_PERTILE_H
#include "kernel_operator.h"
#include "../utils/common_utils.h"
#include "../utils/grouped_matmul_constant.h"
#include "../utils/layout_utils.h"
#include "../utils/tensor_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
#define QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS                                                                         \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeBias_,              \
              typename DataTypeX2Scale_, typename LayoutX1Scale_, typename DataTypeX1Scale_, typename LayoutX2Scale_>
#define QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS                                                                          \
    L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeBias_, DataTypeX2Scale_, LayoutX1Scale_, DataTypeX1Scale_,        \
        LayoutX2Scale_

using namespace Act::Gemm::GroupedMatmul;

struct PerBlockUBParam {
    bool CopyOutWithSplitN = false;
    uint64_t singleM;
    uint64_t singleN;
    uint64_t validM;
    uint64_t validN;
    uint64_t offsetScaleM;
    uint64_t offsetScaleN;
};

namespace {
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t X2SCALE_IDX = 1;
constexpr uint32_t X1SCALE_IDX = 2;
constexpr uint32_t BIAS_IDX = 3;
} // namespace

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
class BlockEpiloguePerTile {
public:
    __aicore__ inline BlockEpiloguePerTile() {}

    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t groupSizeM = 1U;
        uint32_t groupSizeN = 128U;
        uint32_t groupSizeK = 128U;
        Arguments() = default;
    };

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t groupSizeM = 1U;
        uint32_t groupSizeN = 128U;
        uint32_t groupSizeK = 128U;
        Params() = default;
    };

    using YType = DataTypeOut_;
    using CType = DataTypeIn_;
    using BiasType = DataTypeBias_;
    using X2ScaleType = DataTypeX2Scale_;
    using X1ScaleType = DataTypeX1Scale_;
    using LayoutX1Scale = LayoutX1Scale_;
    using LayoutX2Scale = LayoutX2Scale_;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>; // m,n,k
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool transA = TagToTrans<LayoutX1Scale>::value;
    static constexpr bool transB = TagToTrans<LayoutX2Scale>::value;

public:
    __aicore__ inline void Init(const Params* params, AscendC::LocalTensor<CType>* ping,
                                AscendC::LocalTensor<CType>* pong, uint64_t baseL0cSingleV);
    template <class T, bool isFirst>
    __aicore__ inline __ubuf__ T* CopyInX1Scale(uint64_t srcOffset, uint64_t m, uint64_t k,
                                                AscendC::LocalTensor<T>& buf);
    template <class T>
    __aicore__ inline T CopyInX2Scale(__gm__ T* src, uint64_t offset);
    __aicore__ inline void CalcX1OffsetPerGroup();
    __aicore__ inline void CalcX2OffsetPerGroup();
    template <bool isFirstKLoop>
    __aicore__ inline void AivPerTensor(__ubuf__ CType* dst, __ubuf__ CType* l0cOut, __ubuf__ X1ScaleType* x1Scale,
                                        uint16_t mSize, uint16_t nSize, uint16_t kSize, uint32_t nSrcUbAligned,
                                        X2ScaleType x2Scale, uint64_t x1ScaleKIdxInCache);
    __aicore__ inline void AivPostProcess(AscendC::LocalTensor<CType>& mmAddUb);
    __aicore__ inline void operator()(const TupleShape& actualSingleShape, const BlockCoord& blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset);
    __aicore__ inline void UpdateParamsForNextProblem(const TupleShape& problemShape);
    __aicore__ inline void WaitForCube(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreWaitFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_V>(GMM_AIV_SYNC_AIC_FLAG + crossPingPongID);
    }
    __aicore__ inline void NotifyCube(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreSetFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_V>(GMM_AIC_SYNC_AIV_FLAG + crossPingPongID);
    }

private:
    __aicore__ inline void UpdatePerBlockUBValidMN(uint32_t fixpipeN);
    __aicore__ inline void UpdatePerBlockUBParam();
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> vecQueX1Scale_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> vecQueAdd_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> vecQueOut_;

    AscendC::GlobalTensor<YType> cGlobal_;
    AscendC::GlobalTensor<X1ScaleType> x1ScaleGlobal_;
    __gm__ X2ScaleType* x2ScaleGlobal_;
    AscendC::LocalTensor<CType>* mmResPing_;
    AscendC::LocalTensor<CType>* mmResPong_;

private:
    const Params* params_;
    PerBlockUBParam ubParams_;
    TupleShape problemShape_{};
    TupleShape actualSingleShape_{};
    BlockCoord baseOffset_{0, 0, 0, 0};
    BlockCoord blockCoord_{0, 0, 0, 0};

    uint64_t scaleM_ = 0;
    uint64_t scaleN_ = 0;
    uint64_t scaleK_ = 0;
    uint32_t subBlockIdx_;
    uint16_t crossPingPongID_ = 0;
    bool isPertile_ = false;
};

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::Init(
    const Params* params, AscendC::LocalTensor<CType>* ping, AscendC::LocalTensor<CType>* pong, uint64_t baseL0cSingleV)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = params;
    mmResPing_ = ping;
    mmResPong_ = pong;
    pipe_ = GetTPipePtr();

    subBlockIdx_ = AscendC::GetSubBlockIdx();
    pipe_->InitBuffer(vecQueAdd_, 1, baseL0cSingleV * sizeof(CType));
    if constexpr (!AscendC::IsSameType<CType, YType>::value) {
        pipe_->InitBuffer(vecQueOut_, GMM_BUFFER_NUM, Act::Gemm::CeilDiv(baseL0cSingleV * sizeof(YType), 4));
    }

    isPertile_ = params_->groupSizeM == 1;
    if (isPertile_) {
        pipe_->InitBuffer(vecQueX1Scale_, GMM_BUFFER_NUM,
                          Align(Act::Gemm::CeilDiv(Max(params_->baseM, params_->baseN), AscendC::GetTaskRation()) *
                                    GMM_MAX_STEP_SCALEA_K * sizeof(X1ScaleType),
                                GMM_UB_ALIGN_SIZE));
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::UpdateParamsForNextProblem(const TupleShape& problemShape)
{
    problemShape_ = problemShape;

    scaleM_ = Act::Gemm::CeilDiv(Get<MNK_M>(problemShape_), params_->groupSizeM);
    scaleN_ = Act::Gemm::CeilDiv(Get<MNK_N>(problemShape_), params_->groupSizeN);
    scaleK_ = Act::Gemm::CeilDiv(Get<MNK_K>(problemShape_), params_->groupSizeK);
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(const BlockCoord& baseOffset)
{
    if ASCEND_IS_AIV {
        x1ScaleGlobal_.SetGlobalBuffer((__gm__ X1ScaleType*)params_->x1ScaleGmAddr + Get<X1SCALE_IDX>(baseOffset));
        x2ScaleGlobal_ = (__gm__ X2ScaleType*)params_->x2ScaleGmAddr + Get<X2SCALE_IDX>(baseOffset);
        cGlobal_.SetGlobalBuffer((__gm__ YType*)params_->outGmAddr + Get<Y_IDX>(baseOffset));
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CalcX1OffsetPerGroup()
{
    if constexpr (transA) {
        Get<X1SCALE_IDX>(blockCoord_) += ubParams_.offsetScaleM;
    } else {
        Get<X1SCALE_IDX>(blockCoord_) += (ubParams_.offsetScaleM * scaleK_);
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CalcX2OffsetPerGroup()
{
    if constexpr (transB) {
        Get<X2SCALE_IDX>(blockCoord_) += (ubParams_.offsetScaleN * scaleK_);
    } else {
        Get<X2SCALE_IDX>(blockCoord_) += ubParams_.offsetScaleN;
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <class T, bool isFirst>
__aicore__ inline __ubuf__ T*
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CopyInX1Scale(uint64_t srcOffset, uint64_t m, uint64_t k,
                                                                           AscendC::LocalTensor<T>& buf)
{
    AscendC::DataCopyParams x1ScaleGm2UbParams{0, 0, 0, 0};
    AscendC::DataCopyPadParams padParams;
    if constexpr (transA) {
        x1ScaleGm2UbParams.blockCount = k;
        x1ScaleGm2UbParams.blockLen = m * sizeof(T);
        x1ScaleGm2UbParams.srcStride = (scaleM_ - m) * sizeof(T);
    } else {
        x1ScaleGm2UbParams.blockCount = m;
        x1ScaleGm2UbParams.blockLen = k * sizeof(T);
        x1ScaleGm2UbParams.srcStride = (scaleK_ - k) * sizeof(T);
    }
    if constexpr (!isFirst) {
        vecQueX1Scale_.FreeTensor(buf);
        buf = vecQueX1Scale_.template AllocTensor<T>();
    }
    AscendC::DataCopyPad(buf, x1ScaleGlobal_[srcOffset], x1ScaleGm2UbParams, padParams);
    vecQueX1Scale_.EnQue(buf);
    buf = vecQueX1Scale_.template DeQue<T>();
    return reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr());
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <class T>
__aicore__ inline T BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::CopyInX2Scale(__gm__ T* src,
                                                                                               uint64_t offset)
{
    if constexpr (transB) {
        return src[offset];
    } else {
        return src[offset * scaleN_];
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::UpdatePerBlockUBValidMN(uint32_t fixpipeN)
{
    if (ubParams_.CopyOutWithSplitN) {
        ubParams_.validM = ubParams_.singleM;
        uint64_t dirtyN = fixpipeN - Get<MNK_N>(actualSingleShape_);
        if (ubParams_.singleN > dirtyN) {
            if (AscendC::GetSubBlockIdx() == 0) {
                ubParams_.validN = ubParams_.singleN;
            } else {
                ubParams_.validN = ubParams_.singleN - dirtyN;
            }
        } else {
            if (AscendC::GetSubBlockIdx() == 0) {
                ubParams_.validN = Get<MNK_N>(actualSingleShape_);
            } else {
                ubParams_.validN = 0;
            }
        }
    } else {
        if (AscendC::GetSubBlockIdx() == 0) {
            ubParams_.validM = ubParams_.singleM;
        } else {
            ubParams_.validM = Get<MNK_M>(actualSingleShape_) - ubParams_.singleM;
        }
        ubParams_.validN = Get<MNK_N>(actualSingleShape_);
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::UpdatePerBlockUBParam()
{
    ubParams_.CopyOutWithSplitN =
        Get<MNK_N>(actualSingleShape_) > params_->groupSizeN || Get<MNK_M>(actualSingleShape_) == 1;
    uint32_t fixpipeN = 0;
    if (ubParams_.CopyOutWithSplitN) {
        fixpipeN = Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(params_->groupSizeN));
        ubParams_.singleN = fixpipeN / static_cast<uint64_t>(AscendC::GetTaskRation());
    } else {
        fixpipeN = Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        ubParams_.singleN = fixpipeN;
    }
    ubParams_.singleM = ubParams_.CopyOutWithSplitN ? Get<MNK_M>(actualSingleShape_) :
                                                      CeilDiv(Get<MNK_M>(actualSingleShape_), AscendC::GetTaskRation());
    int64_t offsetM = 0;
    int64_t offsetN = 0;
    if (AscendC::GetSubBlockIdx() == 1) {
        if (ubParams_.CopyOutWithSplitN) {
            uint64_t dirtyN = fixpipeN - Get<MNK_N>(actualSingleShape_);
            if (ubParams_.singleN > dirtyN) {
                offsetN = ubParams_.singleN;
            }
        } else {
            offsetM = ubParams_.singleM;
        }
    }
    UpdatePerBlockUBValidMN(fixpipeN);
    ubParams_.offsetScaleM = offsetM;
    ubParams_.offsetScaleN = offsetN / params_->groupSizeN;
    Get<Y_IDX>(blockCoord_) += offsetM * Get<MNK_N>(problemShape_) + offsetN;
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::operator()(const TupleShape& actualSingleShape,
                                                                        const BlockCoord& blockCoord)
{
    actualSingleShape_ = actualSingleShape;
    blockCoord_ = blockCoord;
    UpdatePerBlockUBParam();
    if (subBlockIdx_ == 1) {
        CalcX1OffsetPerGroup();
        CalcX2OffsetPerGroup();
    }
    auto scaleX2Addr = x2ScaleGlobal_ + Get<X2SCALE_IDX>(blockCoord_);

    AscendC::LocalTensor<CType> mmAddUb = vecQueAdd_.template AllocTensor<CType>();
    auto mmAddUbAddr = reinterpret_cast<__ubuf__ CType*>(mmAddUb.GetPhyAddr());
    const uint16_t x1ScaleKElem = Min(GMM_MAX_STEP_SCALEA_K, scaleK_);
    uint64_t kElem;
    __ubuf__ X1ScaleType* x1ScaleUbAddr;
    AscendC::LocalTensor<X1ScaleType> x1ScaleUb = vecQueX1Scale_.template AllocTensor<X1ScaleType>();
    for (uint64_t kb = 0, kOffset = 0; kb < Get<MNK_K>(problemShape_); kb += params_->baseK, kOffset++) {
        X2ScaleType scaleX2 = CopyInX2Scale<X2ScaleType>(scaleX2Addr, kOffset);
        uint64_t x1ScaleKRem = kOffset % x1ScaleKElem;
        if (x1ScaleKRem == 0) {
            kElem = Min(static_cast<uint64_t>(x1ScaleKElem), scaleK_ - kOffset);
            uint64_t scaleX1GmOffset;
            if constexpr (transA) {
                scaleX1GmOffset = Get<X1SCALE_IDX>(blockCoord_) + kOffset * scaleM_;
            } else {
                scaleX1GmOffset = Get<X1SCALE_IDX>(blockCoord_) + kOffset;
            }
            x1ScaleUbAddr = kOffset != 0 ?
                                CopyInX1Scale<X1ScaleType, false>(scaleX1GmOffset, ubParams_.validM, kElem, x1ScaleUb) :
                                CopyInX1Scale<X1ScaleType, true>(scaleX1GmOffset, ubParams_.validM, kElem, x1ScaleUb);
        }

        WaitForCube(crossPingPongID_);
        auto mmUbInputAddr = crossPingPongID_ == 0 ? reinterpret_cast<__ubuf__ CType*>(mmResPing_->GetPhyAddr()) :
                                                     reinterpret_cast<__ubuf__ CType*>(mmResPong_->GetPhyAddr());
        if (kb == 0) {
            AivPerTensor<true>(mmAddUbAddr, mmUbInputAddr, x1ScaleUbAddr, ubParams_.validM, ubParams_.validN, kElem,
                               ubParams_.singleN, scaleX2, x1ScaleKRem);
        } else {
            AivPerTensor<false>(mmAddUbAddr, mmUbInputAddr, x1ScaleUbAddr, ubParams_.validM, ubParams_.validN, kElem,
                                ubParams_.singleN, scaleX2, x1ScaleKRem);
        }
        NotifyCube(crossPingPongID_);
        crossPingPongID_ = (crossPingPongID_ + 1) & 1;
    }
    vecQueX1Scale_.FreeTensor(x1ScaleUb);
    vecQueAdd_.EnQue(mmAddUb);
    mmAddUb = vecQueAdd_.template DeQue<CType>();
    AivPostProcess(mmAddUb);
    vecQueAdd_.FreeTensor(mmAddUb);
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
template <bool isFirstKLoop>
__aicore__ inline void BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::AivPerTensor(
    __ubuf__ CType* dst, __ubuf__ CType* l0cOut, __ubuf__ X1ScaleType* x1Scale, uint16_t mSize, uint16_t nSize,
    uint16_t kSize, uint32_t nSrcUbAligned, X2ScaleType x2Scale, uint64_t x1ScaleKIdxInCache)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(CType);
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    uint16_t alignM = Align(mSize, GMM_UB_ALIGN_SIZE / sizeof(X1ScaleType));
    uint16_t alignK = Align(kSize, GMM_UB_ALIGN_SIZE / sizeof(X1ScaleType));
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            AscendC::MicroAPI::RegTensor<X1ScaleType> x1ScaleReg;
            AscendC::MicroAPI::RegTensor<X1ScaleType> muledScaleReg;
            if constexpr (transA) {
                AscendC::MicroAPI::DataCopy<X1ScaleType, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    x1ScaleReg, x1Scale + x1ScaleKIdxInCache * alignM + mIdx);
            } else {
                AscendC::MicroAPI::DataCopy<X1ScaleType, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    x1ScaleReg, x1Scale + mIdx * alignK + x1ScaleKIdxInCache);
            }
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<CType>(elementNum);
                AscendC::MicroAPI::RegTensor<CType> l0cOutReg;
                AscendC::MicroAPI::RegTensor<CType> addReg;
                AscendC::MicroAPI::RegTensor<CType> ResReg, mulScaleOutReg;
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
                // l0c_out * scale
                AscendC::MicroAPI::Muls(muledScaleReg, x1ScaleReg, x2Scale, maskN);
                AscendC::MicroAPI::Mul(mulScaleOutReg, l0cOutReg, muledScaleReg, maskN);
                uint32_t dstUbOffset = l0cOutOffset;
                if constexpr (isFirstKLoop) {
                    AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                        dst + dstUbOffset, mulScaleOutReg, maskN);
                } else {
                    AscendC::MicroAPI::DataCopy(addReg, dst + l0cOutOffset);
                    AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                    // copy out from register to ub
                    AscendC::MicroAPI::DataCopy<CType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset,
                                                                                                    ResReg, maskN);
                }
            }
        }
    }
}

QGMM_BLOCK_EPILOGUE_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpiloguePerTile<QGMM_BLOCK_EPILOGUE_FUNC_LOCAL_PARAMS>::AivPostProcess(AscendC::LocalTensor<CType>& mmAddUb)
{
    if (ubParams_.validM == 0 || ubParams_.validN == 0) {
        return;
    }
    // Output is transferred to GM in 4 batches
    uint16_t splitNumOfOut = ubParams_.validM >= 4 ? 4 : ubParams_.validM;
    uint64_t mSizeForOnce = Act::Gemm::CeilDiv(ubParams_.validM, static_cast<uint64_t>(splitNumOfOut));
    splitNumOfOut = Act::Gemm::CeilDiv(ubParams_.validM, mSizeForOnce);
    for (uint16_t i = 0; i < splitNumOfOut; i++) {
        uint64_t mLeft = ubParams_.validM - i * mSizeForOnce;
        uint64_t mSize = mLeft >= mSizeForOnce ? mSizeForOnce : mLeft;

        AscendC::LocalTensor<YType> mmRes;
        if constexpr (!AscendC::IsSameType<YType, CType>::value) {
            mmRes = vecQueOut_.template AllocTensor<YType>();
            AscendC::Cast(mmRes, mmAddUb[i * mSizeForOnce * ubParams_.singleN], AscendC::RoundMode::CAST_RINT,
                          mSize * ubParams_.singleN);
            vecQueOut_.EnQue(mmRes);
            mmRes = vecQueOut_.template DeQue<YType>();
        } else {
            mmRes = mmAddUb[i * mSizeForOnce * ubParams_.singleN];
        }

        AscendC::DataCopyExtParams copyParams{0, 0, 0, 0, 0};
        copyParams.blockCount = static_cast<uint16_t>(mSize);
        copyParams.blockLen = static_cast<uint32_t>(ubParams_.validN * sizeof(YType));
        copyParams.srcStride = (ubParams_.singleN - ubParams_.validN) * sizeof(YType) / GMM_DATA_BLOCK;
        copyParams.dstStride = (static_cast<uint64_t>(Get<MNK_N>(problemShape_)) - ubParams_.validN) * sizeof(YType);
        AscendC::DataCopyPad<YType>(cGlobal_[Get<Y_IDX>(blockCoord_) + i * mSizeForOnce * Get<MNK_N>(problemShape_)],
                                    mmRes, copyParams);
        if constexpr (!AscendC::IsSameType<YType, CType>::value) {
            vecQueOut_.FreeTensor(mmRes);
        }
    }
}
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif // EPILOGUE_BLOCK_EPILOGUE_PERTILE_H
