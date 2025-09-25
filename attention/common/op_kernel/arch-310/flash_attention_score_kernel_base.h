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
 * \file flash_attention_score_kernel_base.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_KERNEL_BASE_H_
#define FLASH_ATTENTION_SCORE_KERNEL_BASE_H_
#include "flash_attention_score_block_cube.h"
#include "flash_attention_score_block_vec_train.h"
#include "flash_attention_score_block_vec_infer.h"
#include "kernel_operator.h"
#include "attenmask.h"

// 线上编包
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/CopyInL1.h"

#include "pse.h"
#include "infer_flash_attention_comm.h"
#include "kernel_operator_list_tensor_intf.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
static constexpr uint32_t FA_BYTE_BLOCK = 32;

__aicore__ constexpr uint16_t Align64Func(uint16_t data) {
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreKernelBase {
public:
    ARGS_TRAITS;
    __aicore__ inline FlashAttentionScoreKernelBase() {};

    __aicore__ inline void InitBaseAPI(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                            __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                            __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
                            __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, 
                            __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, 
                            __gm__ uint8_t *deqScaleV, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
                            __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                            __gm__ uint8_t *softmaxOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                            __gm__ uint8_t *workspace, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                            __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                            __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                            __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
                            __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *queryRope,
                            __gm__ uint8_t *keyRope, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
                            __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                            __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace,
                            const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void GetQueryOffset(RunParamStr<isInfer> &runParam);
    __aicore__ inline void GetKeyOffset(RunInfo<isInfer> &runInfo);
    __aicore__ inline void GetKeyRopeOffset(RunInfo<isInfer> &runInfo);
    /* Bmm1右矩阵预取的参数设置 */
    __aicore__ inline void SetTndPrefetchRightArgs(RunInfo<isInfer> &runInfo);
    __aicore__ inline void SetPrefetchRightArgs(RunInfo<isInfer> &runInfo);
    __aicore__ inline uint64_t ComputeNextBatchMorN(RunInfo<isInfer> &runInfo);
    __aicore__ inline void SetS1Base64PrefetchRightArgs(RunInfo<isInfer> &runInfo);
    __aicore__ inline void SetRunInfo(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx, RunParamStr<isInfer> &runParam);
    __aicore__ inline int64_t ComputeNextOffset(RunInfo<isInfer> &runInfo, int64_t nextOffset);
    __aicore__ inline int64_t ComputeNextRopeOffset(RunInfo<isInfer> &runInfo, int64_t nextRopeOffset);
    __aicore__ inline void ComputeBmm1Tail(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam);
    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen);

    __aicore__ inline ChildClass* GetDerived() {
        return static_cast<ChildClass*>(this);
    }
    TPipe *pipe;

    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
    /* 编译期常量的基本块信息 */
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr bool isFp8 = CubeBlockType::isFp8;
    /* 是否使能dn的信息; 没有可选输入并且S2切分的时候使用dn，s2比较小的时候nd效果更好 */
    static constexpr bool useDn = CubeBlockType::useDn;
    static constexpr TPosition bmm2OutPos = CubeBlockType::bmm2OutPos;
    static constexpr bool bmm2Write2Ub = CubeBlockType::bmm2Write2Ub;
    static constexpr bool splitD =  CubeBlockType::splitD;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint64_t SYNC_C1_V1_FLAG[2] = {0, 1};
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr uint64_t SYNC_C2_V2_FLAG[2] = {5, 6};
    /* 核间通道 */
    using bmm2ResGmType = typename std::conditional<bmm2Write2Ub, int32_t, GlobalTensor<float>>::type;
    bmm2ResGmType bmm2ResGm[3];
    TBuf<> bmm2ResBuf[2];
    TBuf<> bmm1ResBuf[2];
    BufferManager<BufferType::L1> l1BufferManager;
    // mm2左矩阵P
    BuffersPolicy3buff<BufferType::L1, false> l1PBuffers; 
    CVSharedParams<isInfer, isPa> sharedParams;
    /* GM信息 */
    using keyGmType = typename std::conditional<isInfer, GlobalTensor<INPUT_T>, int32_t>::type;
    keyGmType keyGm; // kv不连续场景需要使用来获取shape
    __gm__ int64_t *actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlenAddr;
    /* 核Index信息 */
    int32_t aicIdx;

    /* 初始化后不变的信息 */
    ConstInfo<isInfer, hasRope> constInfo;
    AttenMaskInfo attenMaskInfo;
    PseInfo pseInfo;
    /* 其他正向独有的一些信息 */
    // 下一次的key的offset，在预取的场景下可以避免重复计算。
    KVPrefetchArgs<hasRope> prefetchArgs;
    // Unpack参数
    uint64_t s1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitBaseAPI(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    constInfo.subBlockIdx = get_subblockid();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
        this->tilingData = tiling;
    }

    this->pipe = tPipe;
    vecBlock.InitVecBlock(tPipe, this->tilingData, this->sharedParams, this->aicIdx, constInfo.subBlockIdx, 
        attenMaskInfo, pseInfo);
    vecBlock.CleanOutput(softmaxLse, attentionOut, constInfo);
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf();
    if ASCEND_IS_AIC {
        cubeBlock.InitCubeBlock(pipe, &l1BufferManager, query, key, value, blockTable, queryRope, keyRope);
        /* wait kfc message */
        wait_intra_block(PIPE_S, 15);
        auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
        auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
        #pragma unroll
        for (int i = 0; i < sizeof(CVSharedParams<isInfer, isPa>) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
            *tempTiling = *tempTilingSSbuf;
        }
    }

    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, key, value, pse, dropMask, paddingMask, attenMask, prefix,
        actualSeqLengths, actualSeqLengthsKv, deqScaleQ, deqScaleK, deqScaleV, postQuantScale, postQuantOffset,
        queryRope, keyRope, blockTable, queryPaddingSize, kvPaddingSize, softmaxMax, softmaxSum, softmaxOut,
        workspace, tiling, tPipe); // gm设置
    this->InitLocalBuffer();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    // 初始化vector用到的global buffer
    if constexpr (isInfer) {
        keyGm.SetGlobalBuffer((__gm__ INPUT_T *)(key));
        if (constInfo.isQHasLeftPadding) {
            constInfo.queryRightPaddingSize = ((__gm__ int64_t *)queryPaddingSize)[0];
            if (constInfo.queryRightPaddingSize < 0) {
                constInfo.queryRightPaddingSize = 0;
            }
        }
        if (constInfo.isKVHasLeftPadding) {
            constInfo.kvRightPaddingSize = ((__gm__ int64_t *)kvPaddingSize)[0];
            if (constInfo.kvRightPaddingSize < 0) {
                constInfo.kvRightPaddingSize = 0;
            }
        }
    }
    if constexpr (hasAtten) {
        attenMaskInfo.prefixNAddr = prefix;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
        actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
    } else {
        if constexpr (isInfer) {
            if (!constInfo.isActualLenDimsNull) {
                actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
            }
            if (!constInfo.isActualLenDimsKVNull) {
                actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
            }
        }
    }

    if constexpr (!bmm2Write2Ub) {
        int64_t bmm2ResBlock = this->sharedParams.dSizeV;
        if constexpr (splitD) {
            bmm2ResBlock = (int64_t)dVTemplateType;
        }
        int64_t mm2ResultSize = (s1BaseSize) * bmm2ResBlock; // 使用Cube计算的总大小， Gm上的数据按照实际的dSize存储
        int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * sizeof(T);
        int64_t vec2ResultSize = (s1BaseSize) * constInfo.dBasicBlock;
        int64_t vec2Offset = CeilDiv(vec2ResultSize, 128) * 128 * sizeof(T);
        int64_t totalOffset = this->aicIdx * 3 * mm2Offset;
        if constexpr (splitD) {
            totalOffset = this->aicIdx * 3 * (mm2Offset + vec2Offset);
        }
        // SameB模式下V0和V1调用IterateAll的时候填写的地址相同
        this->bmm2ResGm[0].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset));
        this->bmm2ResGm[1].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset + mm2Offset));
        this->bmm2ResGm[2].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset + mm2Offset * 2));
        workspace += (totalOffset + mm2Offset * 3);
    }
    vecBlock.InitGlobalBuffer(pse, deqScaleQ, deqScaleK, deqScaleV, postQuantScale, postQuantOffset,
        prefix, attenMask, dropMask, queryPaddingSize, kvPaddingSize, softmaxMax, softmaxSum, workspace, constInfo);
    cubeBlock.InitCubeInput(key, value, &sharedParams, &attenMaskInfo);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitMMResBuf()
{
    constexpr uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    constexpr uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T); // DYX TODO: CV数量之比暂时定义为宏，后续应根据硬件版本定义为相应常量（1952、David）
    constexpr uint32_t mm2LeftSize = s1BaseSize * s2BaseSize * sizeof(INPUT_T);
    l1BufferManager.Init(pipe, 524288); // 512 * 1024
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);
    this->pipe->InitBuffer(this->bmm1ResBuf[0], mm1ResultSize);
    this->pipe->InitBuffer(this->bmm1ResBuf[1], mm1ResultSize);
    if constexpr (bmm2Write2Ub) {
        this->pipe->InitBuffer(this->bmm2ResBuf[0], mm2ResultSize);
        this->pipe->InitBuffer(this->bmm2ResBuf[1], mm2ResultSize);
    }
}
 
template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    constInfo.s1BaseSize = s1BaseSize;
    constInfo.s2BaseSize = s2BaseSize;
    // 计算轴的乘积

    constInfo.n2Size = sharedParams.n2Size;
    constInfo.s1Size = sharedParams.s1Size;
    constInfo.s2Size = sharedParams.s2Size;
    constInfo.dSize = sharedParams.dSize;
    constInfo.dSizeV = sharedParams.dSizeV;
    constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    if constexpr (hasRope) {
        constInfo.dSizeRope = sharedParams.dSizeRope;
    } else {
        constInfo.dSizeRope = 0;
    }
    constInfo.gSize = sharedParams.gSize;
    constInfo.s1OuterSize = sharedParams.s1OuterSize;
    constInfo.s1D = constInfo.s1Size * constInfo.dSize;
    constInfo.s2D = constInfo.s2Size * constInfo.dSize;
    constInfo.gD = constInfo.gSize * constInfo.dSize;
    constInfo.n2D = constInfo.n2Size * constInfo.dSize;
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    constInfo.bN2D = sharedParams.bSize * constInfo.n2D;
    constInfo.gS1D = constInfo.gSize * constInfo.s1D;
    constInfo.n2S2D = constInfo.n2Size * constInfo.s2D;
    constInfo.n2GD = constInfo.n2Size * constInfo.gD;
    constInfo.bN2GD = sharedParams.bSize * constInfo.n2GD;
    constInfo.n2GS1D = constInfo.n2Size * constInfo.gS1D;
    // 计算切分轴的乘积
    constInfo.s2BaseN2D = s2BaseSize * constInfo.n2D;
    if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
        constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
        constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
        constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
        constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
        constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
        constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
        constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
        constInfo.s2BaseN2Dv = s2BaseSize * constInfo.n2Dv;
        constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;
    } else {
        constInfo.s1Dv = constInfo.s1D;
        constInfo.s2Dv = constInfo.s2D;
        constInfo.n2Dv = constInfo.n2D;
        constInfo.gDv = constInfo.gD;
        constInfo.gS1Dv = constInfo.gS1D;
        constInfo.n2S2Dv = constInfo.n2S2D;
        constInfo.n2GDv = constInfo.n2GD;
        constInfo.s2BaseN2Dv = constInfo.s2BaseN2D;
        constInfo.n2GS1Dv = constInfo.n2GS1D;
    }

    constInfo.layoutType = sharedParams.layoutType;

    if constexpr (hasRope) {
        constInfo.s1DR = constInfo.s1Size * constInfo.dSizeRope;
        constInfo.s2DR = constInfo.s2Size * constInfo.dSizeRope;
        constInfo.gDR = constInfo.gSize * constInfo.dSizeRope;
        constInfo.n2DR = constInfo.n2Size * constInfo.dSizeRope;
        constInfo.bN2DR = sharedParams.bSize * constInfo.n2DR;
        constInfo.gS1DR = constInfo.gSize * constInfo.s1DR;
        constInfo.n2S2DR = constInfo.n2Size * constInfo.s2DR;
        constInfo.n2GDR = constInfo.n2Size * constInfo.gDR;
        constInfo.bN2GDR = sharedParams.bSize * constInfo.n2GDR;
        constInfo.n2GS1DR = constInfo.n2Size * constInfo.gS1DR;
        constInfo.s2BaseN2DR = s2BaseSize * constInfo.n2DR;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        constInfo.s1BaseN2GD = s1BaseSize * constInfo.n2GD;
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
        if constexpr (hasRope) {
            constInfo.s1BaseN2GDR = s1BaseSize * constInfo.n2GDR;
            constInfo.mm1RopeKa = constInfo.n2GDR;
            constInfo.mm1RopeKb = constInfo.n2DR;
        }

        constInfo.mm1Ka = constInfo.n2GD;
        constInfo.mm1Kb = constInfo.n2D;
        constInfo.mm2Kb = constInfo.n2Dv;
        if constexpr (isInfer) {
            if (sharedParams.isGqa) {
                constInfo.mm1Ka = constInfo.dSize;
            }
        }
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
            if constexpr (isInfer) {
                if (sharedParams.isGqa) {
                    constInfo.attentionOutStride = 0;
                }
            }
        }
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSNGD
            constInfo.s1BaseN2GD = s1BaseSize * constInfo.n2GD;
            constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
            if constexpr (hasRope) {
                constInfo.s1BaseN2GDR = s1BaseSize * constInfo.n2GDR;
                constInfo.mm1RopeKa = constInfo.n2GDR;
                constInfo.mm1RopeKb = constInfo.n2DR;
            }
            constInfo.mm1Ka = constInfo.n2GD;
            constInfo.mm1Kb = constInfo.n2D;
            constInfo.mm2Kb = constInfo.n2Dv;
            if constexpr (isInfer) {
                if (sharedParams.isGqa) {
                    constInfo.mm1Ka = constInfo.dSize;
                }
            }
            if ASCEND_IS_AIV {
                constInfo.attentionOutStride =
                    (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
                if constexpr (isInfer) {
                    if (sharedParams.isGqa) {
                        constInfo.attentionOutStride = 0;
                    }
                }
            }
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBNGD
            constInfo.s1BaseBN2GD = s1BaseSize * constInfo.bN2GD;
            constInfo.s2BaseBN2D = sharedParams.bSize * constInfo.s2BaseN2D;
            constInfo.bN2GDv = sharedParams.bSize * constInfo.n2GDv;
            constInfo.s1BaseBN2GDv = s1BaseSize * constInfo.bN2GDv;
            constInfo.s2BaseBN2Dv = sharedParams.bSize * constInfo.s2BaseN2Dv;
            if constexpr (hasRope) {
                constInfo.s1BaseBN2GDR = s1BaseSize * constInfo.bN2GDR;
                constInfo.s2BaseBN2DR = sharedParams.bSize * constInfo.s2BaseN2DR;
                constInfo.mm1RopeKa = constInfo.bN2GDR;
                constInfo.mm1RopeKb = constInfo.bN2DR;
            }
            constInfo.mm1Ka = constInfo.bN2GD;
            constInfo.mm1Kb = constInfo.bN2D;
            constInfo.mm2Kb = sharedParams.bSize * constInfo.n2Dv;
            if ASCEND_IS_AIV {
                constInfo.attentionOutStride =
                    (sharedParams.bSize * constInfo.n2Size * constInfo.gSize - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
            }
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // bnsd
            constInfo.s1BaseD = s1BaseSize * constInfo.dSize;
            constInfo.s2BaseD = s2BaseSize * constInfo.dSize;
            constInfo.s1BaseDv = s1BaseSize * constInfo.dSizeV;
            constInfo.s2BaseDv = s2BaseSize * constInfo.dSizeV;
            if constexpr (hasRope) {
                constInfo.s1BaseDR = s1BaseSize * constInfo.dSizeRope;
                constInfo.s2BaseDR = s2BaseSize * constInfo.dSizeRope;
                constInfo.mm1RopeKa = constInfo.dSizeRope;
                constInfo.mm1RopeKb = constInfo.dSizeRope;
            }
            constInfo.mm1Ka = constInfo.dSize;
            constInfo.mm1Kb = constInfo.dSize;
            constInfo.mm2Kb = constInfo.dSizeV;
            if ASCEND_IS_AIV {
                constInfo.attentionOutStride = 0;
            }
        }
    }
    if ASCEND_IS_AIC {
        if constexpr ((s1BaseSize == 64 && s2BaseSize == 256) || splitD || isFp8) {
            constInfo.enableKVPrefetch = false;
        }
        if constexpr (hasAtten) {
            if (sharedParams.preTokens < constInfo.s1Size) {
                constInfo.enableKVPrefetch = false;
            }
        }
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            if (sharedParams.sparseType == static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
                constInfo.enableKVPrefetch = false;
            }
        }
    }

    if ASCEND_IS_AIV {
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            pseInfo.pseLayoutType = inputParamsRegbase.pseShapeType;
            pseInfo.pseType = inputParamsRegbase.pseType;
            pseInfo.pseBSize = inputParamsRegbase.pseBSize;
            pseInfo.pseS1Size = inputParamsRegbase.pseS1Size;
            pseInfo.pseS2Size = inputParamsRegbase.pseS2Size;
            pseInfo.pseEncodeType = (uint32_t)inputParamsRegbase.pseEncodeType;
            pseInfo.pseStride = pseInfo.pseLayoutType == pse1S2 ? 0 : s2BaseSize;
            pseInfo.qStartIdx = inputParamsRegbase.qStartIdx;
            pseInfo.kvStartIdx = inputParamsRegbase.kvStartIdx;
            if (inputParamsRegbase.pseShapeType == pse1S2) {
                constInfo.gS2 = constInfo.gSize * constInfo.s2Size;
            }
        }

        if constexpr (hasAtten) {
            attenMaskInfo.preTokens = sharedParams.preTokens;
            attenMaskInfo.nextTokens = sharedParams.nextTokens;
            attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
            attenMaskInfo.attenMaskShapeType = inputParamsRegbase.attenMaskShapeType;
            attenMaskInfo.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
            attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
            attenMaskInfo.bandIndex = inputParamsRegbase.bandIndex;
        }
        constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
    }

    GetDerived()->InitUniqueConstInfo();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::Process()
{
    GetDerived()->Process();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetSeqQlenKvlenByBoidx(int64_t boIdx,
    int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
{
    if (unlikely(boIdx == 0)) {
        actualSeqQlen = actualSeqQlenAddr[0];
        actualSeqKvlen = actualSeqKvlenAddr[0];
        return;
    }
    actualSeqQlen = actualSeqQlenAddr[boIdx] - actualSeqQlenAddr[boIdx - 1];
    actualSeqKvlen = actualSeqKvlenAddr[boIdx] - actualSeqKvlenAddr[boIdx - 1];
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::ComputeAxisIdx(
    int64_t multiCoreInnerIdx, RunParamStr<isInfer> &runParam)
{
    // 计算轴的idx
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        GetSeqQlenKvlenByBoidx(runParam.boIdx, runParam.actualS1Size, runParam.actualS2Size);
        int64_t actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(runParam.actualS1Size, this->s1BaseSize) * constInfo.n2G);

        while (multiCoreInnerIdx >= actualS1Outersize) {
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += runParam.actualS1Size;
            this->s2SizeAcc += runParam.actualS2Size;
            runParam.b1SSOffset += runParam.actualS1Size * runParam.actualS2Size;
            if (hasDrop) {
                runParam.b1SSOffsetAlign16 += runParam.actualS1Size * Align(runParam.actualS2Size);
            }
            runParam.boIdx++;
            if (runParam.boIdx >= this->sharedParams.bSize) {
                break;
            }
            GetSeqQlenKvlenByBoidx(runParam.boIdx, runParam.actualS1Size, runParam.actualS2Size);
            actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(runParam.actualS1Size, this->s1BaseSize) * constInfo.n2G);
        }

        int64_t tmpS1Outersize = CeilDiv(runParam.actualS1Size, this->s1BaseSize);
        actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;
        runParam.n2oIdx = actualS1Outersize / tmpS1Outersize / this->sharedParams.gSize;
        runParam.goIdx = actualS1Outersize / tmpS1Outersize % this->sharedParams.gSize;
        runParam.s1oIdx = actualS1Outersize % tmpS1Outersize;
    } else {
        runParam.boIdx = multiCoreInnerIdx / constInfo.n2GS1o;
        runParam.n2oIdx = multiCoreInnerIdx % constInfo.n2GS1o / constInfo.gS1o;
        runParam.goIdx = multiCoreInnerIdx % constInfo.gS1o / constInfo.s1OuterSize;
        runParam.s1oIdx = multiCoreInnerIdx % constInfo.s1OuterSize;
        runParam.b1SSOffset = runParam.boIdx * constInfo.s1S2;
        runParam.actualS1Size = constInfo.s1Size;
        runParam.actualS2Size = constInfo.s2Size;
        if (hasDrop) {
            runParam.b1SSOffsetAlign16 = runParam.boIdx * constInfo.s1Size * Align(constInfo.s2Size);
        }
    }
    runParam.s1RealSize = Min(s1BaseSize, runParam.actualS1Size - runParam.s1oIdx * s1BaseSize);
    if constexpr (useDn) {
        runParam.s1RealSizeAlign32 = (runParam.s1RealSize + 31) >> 5 << 5;
        runParam.halfS1RealSize = runParam.s1RealSize <= 16 ? runParam.s1RealSize : (runParam.s1RealSizeAlign32 >> 1);
    } else {
        runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
    }
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    runInfo.s2StartIdx = runParam.s2LineStartIdx;
    runInfo.s2LoopStartIdx = runParam.s2LoopStartIdx;
    runInfo.s2EndIdx = runParam.s2LineEndIdx;
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        runInfo.boIdx = runParam.boIdx;
        runInfo.s1SizeAcc = s1SizeAcc;
        runInfo.s2SizeAcc = s2SizeAcc;
    } else {
        runInfo.s2SizeAcc = runInfo.boIdx * constInfo.s2Size;
    }
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    if constexpr (isFd) {
        runInfo.flashDecodeS2Idx = this->aicIdx % constInfo.splitKVNum;
    }
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.queryOffset = runParam.tensorQOffset;
    runInfo.qRopeOffset = runParam.qRopeNBGOffset;
    this->ComputeBmm1Tail(runInfo, runParam);
    GetDerived()->InitUniqueRunInfo(runParam, runInfo);
    GetKeyOffset(runInfo);
    if constexpr (hasRope) {
        GetKeyRopeOffset(runInfo);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.s1RealSizeAlign32 = runParam.s1RealSizeAlign32;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;

    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;  // D>128 这里需要适配

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    if constexpr (isInfer) {
        if ((runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
            runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize;
            runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
        }
    } else {
        if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
            runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
            runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetQueryOffset(
    RunParamStr<isInfer> &runParam)
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    // s1需要考虑inner轴的影响
    int64_t s1Offset = 0;

    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t bOffsetOut = 0;
    int64_t s1OffsetOut = 0;
    int64_t n2OffsetOut = 0;
    int64_t gOffsetOut = 0;
    int64_t subBlockS1Offset = 0;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        bOffset = this->s1SizeAcc * constInfo.n2GD;
        s1Offset = runParam.s1oIdx * constInfo.s1BaseN2GD;
        n2Offset = runParam.n2oIdx * constInfo.gD;
        gOffset = runParam.goIdx * constInfo.dSize;
        bOffsetOut = this->s1SizeAcc * constInfo.n2GDv;
        s1OffsetOut = runParam.s1oIdx * constInfo.s1BaseN2GDv;
        n2OffsetOut = runParam.n2oIdx * constInfo.gDv;
        gOffsetOut = runParam.goIdx * constInfo.dSizeV;
        subBlockS1Offset = constInfo.subBlockIdx * runParam.firstHalfS1RealSize * constInfo.n2GDv;
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSNGD
            bOffset = runParam.boIdx * constInfo.n2GS1D;
            s1Offset = runParam.s1oIdx * constInfo.s1BaseN2GD;
            n2Offset = runParam.n2oIdx * constInfo.gD;
            gOffset = runParam.goIdx * constInfo.dSize;
            bOffsetOut = runParam.boIdx * constInfo.n2GS1Dv;
            s1OffsetOut = runParam.s1oIdx * constInfo.s1BaseN2GDv;
            n2OffsetOut = runParam.n2oIdx * constInfo.gDv;
            gOffsetOut = runParam.goIdx * constInfo.dSizeV;
            subBlockS1Offset = constInfo.subBlockIdx * runParam.firstHalfS1RealSize * constInfo.n2GDv;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBNGD
            s1Offset = runParam.s1oIdx * constInfo.s1BaseBN2GD;
            bOffset = runParam.boIdx * constInfo.n2GD;
            n2Offset = runParam.n2oIdx * constInfo.gD;
            gOffset = runParam.goIdx * constInfo.dSize;
            s1OffsetOut = runParam.s1oIdx * constInfo.s1BaseBN2GDv;
            bOffsetOut = runParam.boIdx * constInfo.n2GDv;
            n2OffsetOut = runParam.n2oIdx * constInfo.gDv;
            gOffsetOut = runParam.goIdx * constInfo.dSizeV;
            subBlockS1Offset = constInfo.subBlockIdx * runParam.firstHalfS1RealSize * constInfo.bN2GDv;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // bnsd
            bOffset = runParam.boIdx * constInfo.n2GS1D;
            n2Offset = runParam.n2oIdx * constInfo.gS1D;
            gOffset = runParam.goIdx * constInfo.s1D;
            s1Offset = runParam.s1oIdx * constInfo.s1BaseD;
            bOffsetOut = runParam.boIdx * constInfo.n2GS1Dv;
            n2OffsetOut = runParam.n2oIdx * constInfo.gS1Dv;
            gOffsetOut = runParam.goIdx * constInfo.s1Dv;
            s1OffsetOut = runParam.s1oIdx * constInfo.s1BaseDv;
            subBlockS1Offset = constInfo.subBlockIdx * runParam.firstHalfS1RealSize * constInfo.dSizeV;
        }
    }
    if ASCEND_IS_AIC {
        runParam.tensorQOffset = bOffset + n2Offset + gOffset + s1Offset;
    } else {
        runParam.attentionOutOffset = bOffsetOut + n2OffsetOut + gOffsetOut + s1OffsetOut + subBlockS1Offset;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetTndPrefetchRightArgs(
    RunInfo<isInfer> &runInfo)
{
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        int64_t boIdxNext = runInfo.boIdx;
        int64_t n2oIdxNext = runInfo.n2oIdx;
        uint32_t endS1Loop = CeilDiv(runInfo.actualS1Size, s1BaseSize) - 1;
        if (runInfo.s1oIdx == endS1Loop) {
            if (runInfo.goIdx == constInfo.gSize - 1) {
                n2oIdxNext++;
                if (n2oIdxNext == constInfo.n2Size) {
                    boIdxNext++;
                    n2oIdxNext = 0;
                }
            }
        }
        int64_t bSize = this->sharedParams.bSize;
        if (boIdxNext == bSize) {
            return; // 最后一个batch的最后一次s2，预取应该不使能
        }
        int64_t nextActualS2Len = runInfo.actualS2Size;
        // 找到下一个有效的batch
        if (boIdxNext != runInfo.boIdx) {
            int64_t nextActualS1Len;
            for (; boIdxNext < bSize; boIdxNext++) {
                GetSeqQlenKvlenByBoidx(boIdxNext, nextActualS1Len, nextActualS2Len);
                if (nextActualS1Len != 0 && nextActualS2Len != 0) {
                    break;
                }
            }
        }

        if (nextActualS2Len >= s2BaseSize) {
            prefetchArgs.nextMOrN = s2BaseSize;
        } else {
            prefetchArgs.nextMOrN = nextActualS2Len % s2BaseSize;
            if (prefetchArgs.nextMOrN == 0) {
                prefetchArgs.nextMOrN = s2BaseSize;
            }
        }
        if (boIdxNext == 0) {
            prefetchArgs.nextOffset = n2oIdxNext * constInfo.dSize + runInfo.s2StartIdx * constInfo.n2D;
            if constexpr (hasRope) {
                prefetchArgs.nextRopeOffset = n2oIdxNext * constInfo.dSizeRope + runInfo.s2StartIdx * constInfo.n2DR;
            }
        } else {
            // runInfo.s2StartIdx是为了推理保留，训练应该不需要，推理左padding场景下，需要获取下一个batch的左padding值
            int32_t nextTotalActualS2Len = actualSeqKvlenAddr[boIdxNext - 1];
            prefetchArgs.nextOffset = nextTotalActualS2Len * constInfo.n2D + n2oIdxNext * constInfo.dSize +
                            runInfo.s2StartIdx * constInfo.n2D;
            if constexpr (hasRope) {
                prefetchArgs.nextRopeOffset = nextTotalActualS2Len * constInfo.n2DR + n2oIdxNext * constInfo.dSizeRope +
                                    runInfo.s2StartIdx * constInfo.n2DR;
            }
        }
    } else {
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit - 1) {
            prefetchArgs.nextMOrN = s2BaseSize;
        } else {
            prefetchArgs.nextMOrN = (runInfo.s2EndIdx - runInfo.s2StartIdx) % s2BaseSize;
            if (prefetchArgs.nextMOrN == 0) {
                prefetchArgs.nextMOrN = s2BaseSize;
            }
        }
        prefetchArgs.nextOffset += constInfo.s2BaseN2D;
        if constexpr (hasRope) {
            prefetchArgs.nextRopeOffset += constInfo.s2BaseN2DR;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetPrefetchRightArgs(
    RunInfo<isInfer> &runInfo) {
    if constexpr (!isInfer && layout == LayOutTypeEnum::LAYOUT_TND) {
        SetTndPrefetchRightArgs(runInfo);
    } else {
        if constexpr (s1BaseSize == 128) {
            if (runInfo.s2LoopLimit > 0 && (runInfo.s2LoopCount != runInfo.s2LoopLimit - 1)) {
                // 这种情况一定不是尾块
                prefetchArgs.nextMOrN = s2BaseSize;
            } else {
                prefetchArgs.nextMOrN = (runInfo.s2EndIdx - runInfo.s2StartIdx) % s2BaseSize;
                if (prefetchArgs.nextMOrN == 0) {
                    prefetchArgs.nextMOrN = s2BaseSize;
                }
            }
        } else {
            SetS1Base64PrefetchRightArgs(runInfo);
        }

        prefetchArgs.nextOffset = ComputeNextOffset(runInfo, prefetchArgs.nextOffset);
        if constexpr (hasRope) {
            prefetchArgs.nextRopeOffset = ComputeNextRopeOffset(runInfo, prefetchArgs.nextRopeOffset);
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline uint64_t FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::ComputeNextBatchMorN(
    RunInfo<isInfer> &runInfo) {
    int64_t s2NextStartIdx = 0;
    int64_t s2NextEndIdx = 0;
    if (this->sharedParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) {
        s2NextStartIdx = 0;
        s2NextEndIdx = Min(s1BaseSize, s2BaseSize);
    } else if (this->sharedParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
        s2NextStartIdx = Max(0 - this->sharedParams.s1SparseValidSize, 0);
        s2NextEndIdx = Min(s1BaseSize + this->sharedParams.s2SparseValidSize,
                            s2BaseSize);
    } else if (this->sharedParams.sparseType == static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
        // TODO: 暂时删除Prefix场景的预取，保证CubeBlock的纯粹性
        s2NextStartIdx = 0;
        if (unlikely(runInfo.boIdx >= this->sharedParams.bSize - 1)) {
            s2NextEndIdx = 0;
        } else {
            s2NextEndIdx = Max(s1BaseSize - constInfo.s1Size + constInfo.s2Size,
                               ((__gm__ int64_t *)attenMaskInfo.prefixNAddr)[runInfo.boIdx + 1]);
            s2NextEndIdx = CeilDiv(s2NextEndIdx, s2BaseSize) * s2BaseSize;
            s2NextEndIdx = Min(s2NextEndIdx, s2BaseSize);
        }
    } else {
        s2NextStartIdx = 0;
        s2NextEndIdx = s2BaseSize;
    }
    return s2NextEndIdx - s2NextStartIdx;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetS1Base64PrefetchRightArgs(
    RunInfo<isInfer> &runInfo) {
    if (runInfo.s2LoopLimit > 0 && (runInfo.s2LoopCount != runInfo.s2LoopLimit - 1)) {
        if constexpr (hasAtten) {
            if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit && runInfo.s1oIdx == constInfo.s1OuterSize - 1)) {
                prefetchArgs.nextMOrN = ComputeNextBatchMorN(runInfo);
            } else {
                prefetchArgs.nextMOrN = s2BaseSize;
            }
        } else {
            prefetchArgs.nextMOrN = s2BaseSize;
        }
    } else if (runInfo.s2LoopLimit == 0) {
        prefetchArgs.nextMOrN = Min(s2BaseSize, constInfo.s2Size);
    } else {
        prefetchArgs.nextMOrN = (runInfo.s2EndIdx - runInfo.s2StartIdx) % s2BaseSize;
    }
    if (prefetchArgs.nextMOrN == 0) {
        prefetchArgs.nextMOrN = s2BaseSize;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetKeyOffset(
    RunInfo<isInfer> &runInfo)
{
    if ASCEND_IS_AIC {
        if (constInfo.enableKVPrefetch && runInfo.taskId != 0) {
            runInfo.keyOffset = prefetchArgs.nextOffset;
            if constexpr (!isInfer) {
                runInfo.valueOffset = runInfo.keyOffset;
            }
        } else {
            if constexpr (isInfer) {
                prefetchArgs.nextOffset = runInfo.keyOffset;
            } else {
                // 计算gm上的offset
                int64_t bOffset = 0;
                int64_t n2Offset = 0;
                int64_t s2Offset = 0;

                if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
                    // (BS)ND
                    bOffset = runInfo.s2SizeAcc * constInfo.n2D;
                    s2Offset = runInfo.s2StartIdx * constInfo.n2D + runInfo.s2LoopCount * constInfo.s2BaseN2D;
                    n2Offset = runInfo.n2oIdx * constInfo.dSize;
                } else {
                    if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
                        // BSH/BSND
                        bOffset = runInfo.boIdx * constInfo.n2S2D;
                        s2Offset = runInfo.s2StartIdx * constInfo.n2D + runInfo.s2LoopCount * constInfo.s2BaseN2D;
                        n2Offset = runInfo.n2oIdx * constInfo.dSize;
                    } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
                        // SBH/SBND
                        s2Offset = runInfo.s2StartIdx * constInfo.bN2D + runInfo.s2LoopCount * constInfo.s2BaseBN2D;
                        bOffset = runInfo.boIdx * constInfo.n2D;
                        n2Offset = runInfo.n2oIdx * constInfo.dSize;
                    } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
                        // BNSD
                        bOffset = runInfo.boIdx * constInfo.n2S2D;
                        n2Offset = runInfo.n2oIdx * constInfo.s2D;
                        s2Offset = runInfo.s2StartIdx * constInfo.dSize + runInfo.s2LoopCount * constInfo.s2BaseD;
                    }
                }
                runInfo.keyOffset = bOffset + n2Offset + s2Offset;
                runInfo.valueOffset = runInfo.keyOffset;
                prefetchArgs.nextOffset = runInfo.keyOffset;
            }
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetKeyRopeOffset(
    RunInfo<isInfer> &runInfo)
{
    if ASCEND_IS_AIC {
        if (constInfo.enableKVPrefetch && runInfo.taskId != 0) {
            runInfo.kRopeOffset = prefetchArgs.nextRopeOffset;
        }

        if constexpr (isInfer) {
            prefetchArgs.nextRopeOffset = runInfo.kRopeOffset;
        } else {
            // 计算gm上的offset
            int64_t bOffsetRope = 0;
            int64_t n2OffsetRope = 0;
            int64_t s2OffsetRope = 0;

            if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
                // (BS)ND
                bOffsetRope = runInfo.s2SizeAcc * constInfo.n2DR;
                s2OffsetRope = runInfo.s2StartIdx * constInfo.n2DR + runInfo.s2LoopCount * constInfo.s2BaseN2DR;
                n2OffsetRope = runInfo.n2oIdx * constInfo.dSizeRope;
            } else {
                if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
                    // BSH/BSND
                    bOffsetRope = runInfo.boIdx * constInfo.n2S2DR;
                    s2OffsetRope = runInfo.s2StartIdx * constInfo.n2DR + runInfo.s2LoopCount * constInfo.s2BaseN2DR;
                    n2OffsetRope = runInfo.n2oIdx * constInfo.dSizeRope;
                } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
                    // SBH/SBND
                    s2OffsetRope = runInfo.s2StartIdx * constInfo.bN2DR + runInfo.s2LoopCount * constInfo.s2BaseBN2DR;
                    bOffsetRope = runInfo.boIdx * constInfo.n2DR;
                    n2OffsetRope = runInfo.n2oIdx * constInfo.dSizeRope;
                } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
                    // BNSD
                    bOffsetRope = runInfo.boIdx * constInfo.n2S2DR;
                    n2OffsetRope = runInfo.n2oIdx * constInfo.s2DR;
                    s2OffsetRope = runInfo.s2StartIdx * constInfo.dSizeRope +
                        runInfo.s2LoopCount * constInfo.s2BaseDR;
                }
            }
            runInfo.kRopeOffset = bOffsetRope + n2OffsetRope + s2OffsetRope;
            prefetchArgs.nextRopeOffset = runInfo.kRopeOffset;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::ComputeNextOffset(
    RunInfo<isInfer> &runInfo, int64_t nextOffset)
{
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        int64_t multiCoreInnerIdxNext = runInfo.multiCoreInnerIdx + 1;
        int64_t boIdxNext = runInfo.boIdx;
        int64_t n2oIdxNext = runInfo.n2oIdx;
        if constexpr (isInfer) {
            uint32_t endS1Loop = CeilDivision(runInfo.actualS1Size, (int64_t)s1BaseSize) - 1;
            int64_t gIdxNext = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx;
            if (runInfo.s1oIdx == endS1Loop) {
                gIdxNext++;
                if (runInfo.goIdx == constInfo.gSize - 1) {
                    n2oIdxNext++;
                }
            }
            if (gIdxNext == constInfo.n2G) {
                boIdxNext++;
                n2oIdxNext = 0;
            }
        } else {
            boIdxNext = multiCoreInnerIdxNext / constInfo.n2GS1o;
            n2oIdxNext = multiCoreInnerIdxNext % constInfo.n2GS1o / constInfo.gS1o;
        }

        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSND
            nextOffset = boIdxNext * constInfo.n2S2D + n2oIdxNext * constInfo.dSize +
                runInfo.s2StartIdx * constInfo.n2D;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBND
            nextOffset = boIdxNext * constInfo.n2D + n2oIdxNext * constInfo.dSize +
                runInfo.s2StartIdx * constInfo.bN2D;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // BNSD
            nextOffset = boIdxNext * constInfo.n2S2D + n2oIdxNext * constInfo.s2D +
                runInfo.s2StartIdx * constInfo.dSize;
        } else if (layout == LayOutTypeEnum::LAYOUT_TND) {
            // TND
            if constexpr (isInfer) {
                if (boIdxNext == 0) {
                    prefetchArgs.nextOffset = n2oIdxNext * constInfo.dSize + runInfo.s2StartIdx * constInfo.n2D;
                } else {
                    prefetchArgs.nextOffset = actualSeqKvlenAddr[boIdxNext - 1] * constInfo.n2D +
                        n2oIdxNext * constInfo.dSize + runInfo.s2StartIdx * constInfo.n2D;
                }
            }
        }
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSND
            nextOffset += constInfo.s2BaseN2D;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBND
            nextOffset += constInfo.s2BaseBN2D;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // BNSD
            nextOffset += constInfo.s2BaseD;
        } else if (layout == LayOutTypeEnum::LAYOUT_TND) {
            nextOffset += constInfo.s2BaseN2D;
        }
    }
    return nextOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::ComputeNextRopeOffset(
    RunInfo<isInfer> &runInfo, int64_t nextRopeOffset)
{
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        int64_t multiCoreInnerIdxNext = runInfo.multiCoreInnerIdx + 1;
        int64_t boIdxNext = runInfo.boIdx;
        int64_t n2oIdxNext = runInfo.n2oIdx;
        if constexpr (isInfer) {
            uint32_t endS1Loop = CeilDivision(runInfo.actualS1Size, (int64_t)s1BaseSize) - 1;
            int64_t gIdxNext = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx;
            if (runInfo.s1oIdx == endS1Loop) {
                gIdxNext++;
                if (runInfo.goIdx == constInfo.gSize - 1) {
                    n2oIdxNext++;
                }
            }
            if (gIdxNext == constInfo.n2G) {
                boIdxNext++;
                n2oIdxNext = 0;
            }
        } else {
            boIdxNext = multiCoreInnerIdxNext / constInfo.n2GS1o;
            n2oIdxNext = multiCoreInnerIdxNext % constInfo.n2GS1o / constInfo.gS1o;
        }
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSND
            nextRopeOffset = boIdxNext * constInfo.n2S2DR + n2oIdxNext * constInfo.dSizeRope +
                runInfo.s2StartIdx * constInfo.n2DR;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBND
            nextRopeOffset = boIdxNext * constInfo.n2DR + n2oIdxNext * constInfo.dSizeRope +
                runInfo.s2StartIdx * constInfo.bN2DR;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // BNSD
            nextRopeOffset = boIdxNext * constInfo.n2S2DR + n2oIdxNext * constInfo.s2DR +
                runInfo.s2StartIdx * constInfo.dSizeRope;
        } else if (layout == LayOutTypeEnum::LAYOUT_TND) {
            // TND
            if constexpr (isInfer) {
                if (boIdxNext == 0) {
                    nextRopeOffset = n2oIdxNext * constInfo.dSizeRope +
                        runInfo.s2StartIdx * constInfo.n2DR;
                } else {
                    nextRopeOffset = actualSeqKvlenAddr[boIdxNext - 1] * constInfo.n2DR +
                        n2oIdxNext * constInfo.dSizeRope + runInfo.s2StartIdx * constInfo.n2DR;
                }
            }
        }
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSND
            nextRopeOffset += constInfo.s2BaseN2DR;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBND
            nextRopeOffset += constInfo.s2BaseBN2DR;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // BNSD
            nextRopeOffset += constInfo.s2BaseDR;
        } else if (layout == LayOutTypeEnum::LAYOUT_TND) {
            // TND
            nextRopeOffset += constInfo.s2BaseN2DR;
        }
    }
    return nextRopeOffset;
}
}
#endif // FLASH_ATTENTION_SCORE_KERNEL_BASE_H_