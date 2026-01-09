/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attn_sharedkv_swa.h
 * \brief
 */

#ifndef SPARSE_ATTN_SHAREDKV_SWA_H
#define SPARSE_ATTN_SHAREDKV_SWA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_attn_sharedkv_common.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SAST> class SparseAttnSharedkvSwa {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SAST::queryType;
    // using KV_T = typename SAST::kvType;
    using OUT_T = typename SAST::outputType;
    using UPDATE_T = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;

    __aicore__ inline SparseAttnSharedkvSwa(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
                                __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *sinks,
                                __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *seqUsedKV,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const SparseAttnSharedkvTilingDataSwa *__restrict tiling,
				                __gm__ uint8_t *gmTiling, TPipe *tPipe);

    __aicore__ inline void Process();

private:
    const SparseAttnSharedkvTilingDataSwa *__restrict tilingData = nullptr;

    TPipe *pipe = nullptr;

    uint64_t mSizeVStart = 0ULL;
    int64_t threshold = 0;
    uint64_t topKBaseOffset = 0ULL;
    uint64_t s2BatchBaseOffset = 0;
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBRopeCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    // ConstInfo constInfo{};
    // TempLoopInfo tempLoopInfo{};

    // SFAMatmulService<SAST> matmulService;
    // SFAVectorService<SAST> vectorService;

    // GlobalTensor<Q_T> queryGm;
    // GlobalTensor<KV_T> keyGm;
    // GlobalTensor<KV_T> valueGm;

    GlobalTensor<OUT_T> attentionOutGm;
    // GlobalTensor<int32_t> blockTableGm;
    // GlobalTensor<int32_t> topKGm;

    // GlobalTensor<int32_t> actualSeqLengthsQGm;
    // GlobalTensor<int32_t> actualSeqLengthsKVGm;

    // // workspace
    // GlobalTensor<MM1_OUT_T> mm1ResGm;
    // GlobalTensor<KV_T> vec1ResGm;
    // GlobalTensor<MM2_OUT_T> mm2ResGm;
    // GlobalTensor<KV_T> kvMergeGm_;

    // GlobalTensor<int32_t> mm2ResInt32Gm;
    // GlobalTensor<UPDATE_T> vec2ResGm;

    // ================================Init functions===================================
    __aicore__ inline void InitTilingData();
    // __aicore__ inline void InitCalcParamsEach();
    // __aicore__ inline void InitBuffers();
    // __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    // __aicore__ inline void InitOutputSingleCore();
    // // ================================Process functions================================
    // __aicore__ inline void ProcessBalance();
    // __aicore__ inline void PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
    //                                        RunInfo extraInfo[SFA_PRELOAD_TASK_CACHE_SIZE], uint32_t &curTopKIdx, uint64_t &curOffsetInSparseBlock);
    // // ================================Offset Calc=====================================
    // __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    // __aicore__ inline void GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    // __aicore__ inline void CalcSinnerTopKBegin(RunInfo &info, uint32_t &curTopKIdx, uint64_t &curOffsetInSparseBlock);
    // __aicore__ inline void UpdateInnerLoopCond();
    // __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    // __aicore__ inline void CalcParams(uint32_t loop, uint64_t s2Start, uint32_t s2LoopIdx, RunInfo &info);
    // __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t gS1EndPrev, uint32_t s2EndPrev);
    // __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx);
    // __aicore__ inline uint32_t GetActualSeqLenKV(uint32_t bIdx);
    // __aicore__ inline void GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx);
    // __aicore__ inline void UpdateInner(uint32_t &s2End, uint32_t &curS2End, uint32_t s1Idx, bool isEnd);
    // __aicore__ inline void GetPreNextTokensLeftUp();
    // // ================================Mm1==============================================
    // __aicore__ inline void ComputeMm1(const RunInfo &info);
    // // ================================Mm2==============================================
    // __aicore__ inline void ComputeMm2(const RunInfo &info);
    // __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
    //                                        uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    // __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
};

template <typename SAST> __aicore__ inline void SparseAttnSharedkvSwa<SAST>::InitTilingData()
{
//     usedCoreNum = tilingData->singleCoreParams.usedCoreNum;

}

// template <typename SAST> __aicore__ inline void SparseAttnSharedkvSwa<SAST>::InitBuffers()
// {
//     if ASCEND_IS_AIV {
//         vectorService.InitBuffers(pipe);
//     } else {
//         matmulService.InitBuffers(pipe);
//     }
// }

template <typename SAST>
__aicore__ inline void SparseAttnSharedkvSwa<SAST>::Init(
                        __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                        __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
                        __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *sinks,
                        __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *seqUsedKV,
                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                        const SparseAttnSharedkvTilingDataSwa *__restrict tiling,
                        __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // init tiling data
    tilingData = tiling;

    InitTilingData();
    // InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);

    // 初始化计算参数
    // InitCalcParamsEach();
    // pipe = tPipe;
    // keyPtr = key;
    // valuePtr = value;

    // // init global buffer
    // queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    // keyGm.SetGlobalBuffer((__gm__ KV_T *)keyPtr);
    // valueGm.SetGlobalBuffer((__gm__ KV_T *)valuePtr);
    // qRopeGm.SetGlobalBuffer((__gm__ Q_ROPE_T *)queryRope);
    // kRopeGm.SetGlobalBuffer((__gm__ K_ROPE_T *)keyRope);

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);

}


template <typename SAST> __aicore__ inline void SparseAttnSharedkvSwa<SAST>::Process()
{
    // if (aiCoreIdx < usedCoreNum) {
    //     if ASCEND_IS_AIV {
    //         vectorService.AllocEventID();
    //         vectorService.InitSoftmaxDefaultBuffer();
    //     } else {
    //         matmulService.AllocEventID();
    //     }
    //     ProcessBalance();

    //     if ASCEND_IS_AIV {
    //         vectorService.FreeEventID();
    //     } else {
    //         matmulService.FreeEventID();
    //     }
    // }
}

#endif // SPARSE_ATTN_SHAREDKV_SWA_H