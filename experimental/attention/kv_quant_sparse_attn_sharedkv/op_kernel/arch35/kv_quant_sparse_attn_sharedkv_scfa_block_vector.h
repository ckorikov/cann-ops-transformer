/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_block_vec_base_scfa.h
 * \brief
 */
 // TODO 修改
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H

#include "util_regbase.h"
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"
#include "common/buffers_policy.h"
#include "common/buffer_manager.h"
#include "common/buffer.h"
#include "kernel_operator_list_tensor_intf.h"

#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_scfa.h"
#include "vf/vf_flashupdate_new_scfa.h"

using namespace AscendC;
using namespace SCFaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace optiling;
using namespace optiling::detail;
using namespace regbaseutil;

namespace BaseApi {
TEMPLATES_DEF
class SCFABlockVec {
public:
    /* =================编译期常量的基本块信息================= */
    // TODO 来自模板参数 后续修改
    // static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    // static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1HalfS1BaseSize = s1BaseSize >> 1;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;

    // TODO 确认传参自哪里
    // static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)512);
    SasMetaData metadataVecLocal;

    // ==================== Functions ======================
    __aicore__ inline SCFABlockVec() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseAttnSharedkvTilingData *__restrict tiling,
        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, SasMetaData &metadataLocal) {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            tilingData = tiling;
            metadataVecLocal = metadataLocal;
            this->InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *cmpSparseIndices,
        __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv,
        __gm__ uint8_t *sinks);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);

    // ==================== Vector0 ======================
    __aicore__ inline void MergeKv(const RunInfo &runInfo);
    __aicore__ inline void MergeOriKv(const RunInfo &runInfo);
    __aicore__ inline int64_t GetKeyBNBOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &runInfo);
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo &runInfo);
    __aicore__ inline void DequantKv(int64_t mergeMte3Idx, int64_t dealRow);
    __aicore__ inline void CopyOutKvUb2L1(int64_t mergeMte3Idx, int64_t nopeGmOffset, int64_t ropeGmOffset, int64_t dealRow);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo);
    // ==================== Vector1 ======================
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

    TPipe *tPipe;
    // V0
    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;
    // V0
    const KvQuantSparseAttnSharedkvTilingData *__restrict tilingData;
    
    // 在CleanOutput中初始化
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<Q_T> sinksGm;
    __gm__ int64_t *actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlenAddr;

    /* =====================V侧UB变量==================== */
    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2]; 
    /* =================初始化后不变的信息================= */
    T negativeFloatScalar;

protected:
/* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    using VEC2_RES_T = float;
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize);

private:
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void GetExtremeValue(T &negativeScalar);

    // for V0
    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_512B / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);  // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                                // 32/4=8
    static constexpr uint32_t LIMIT_DEAL_ROW = 16U;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<Q_T> kvMergeGm_; // 改成L1
    GlobalTensor<Q_T> keyRopeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;

    // ================================Local Buffer区====================================
    TBuf<> inputBuff2;  // 32K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 4K

    TBuf<> tmpBuff1;         // 32K
    TBuf<> tmpBuff2;         // 8K
    TBuf<> v0ValidSizeBuff;  // 8K

    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                              int64_t topkGmBaseOffset, const RunInfo &runInfo)
{
    int64_t topkGmIdx = (s2GmOffset + runInfo.s2LoopCount * constInfo.s2BaseSize) / constInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= constInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + runInfo.s2LoopCount * constInfo.s2BaseSize) % constInfo.sparseBlockSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t SCFABlockVec<TEMPLATE_ARGS>::GetKeyBNBOffset(int64_t realS2Idx,
                                                                    const RunInfo &runInfo, int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyBNBOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = realS2Idx / constInfo.blockSize;
        int64_t blkTableOffset = realS2Idx % constInfo.blockSize;
        realKeyBNBOffset = blkTableGm_.GetValue(runInfo.boIdx * constInfo.maxBlockNumPerBatch + blkTableIdx) *
                                static_cast<int64_t>(constInfo.blockSize) *
                                static_cast<int64_t>(constInfo.n2Size) +
                                blkTableOffset;
    } else {
        // realKeyBNBOffset = (runInfo.tensorBOffset +
        realKeyBNBOffset = (runInfo.keyOffset + // todo check if keyOffset is tensorBOffset
                           realS2Idx * constInfo.n2Size * constInfo.dSizeCombine) /
                           constInfo.dSizeCombine;
    }
    return realKeyBNBOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
SCFABlockVec<TEMPLATE_ARGS>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                         int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;
    
    intriParams.blockCount = validS2Count;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    // if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
        uint32_t combineBytes = (constInfo.dSize * sizeof(KV_T) + constInfo.dSizeRope * sizeof(Q_T) +
            constInfo.dSize / constInfo.tileSize * sizeof(T));
        intriParams.blockLen = combineBytes;
        uint32_t combineDim = combineBytes / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)  + (mte2Size - mte3Size) *
                combineDimAlign], keyGm_[keyBNBOffset * combineDim], intriParams, padParams);
    // }
    mte2Size += validS2Count;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                          int64_t realS2Idx1, int64_t realS2Idx2,
                                                          const RunInfo &runInfo)
{
    int64_t s2IdLimit = runInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        s2IdLimit = runInfo.curActualSeqLenOri - runInfo.actualS1Size + runInfo.gS1Idx / constInfo.gSize + 1; // todo del gS1Idx
    }

    int64_t keyBNBOffset1 = GetKeyBNBOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyBNBOffset2 = GetKeyBNBOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyBNBOffset1 < 0 && keyBNBOffset2 < 0)) {
        return;
    }

    int64_t sparseBlockSrcStride =
        ((keyBNBOffset1 > keyBNBOffset2 ? (keyBNBOffset1 - keyBNBOffset2) :
        (keyBNBOffset2 - keyBNBOffset1)) - constInfo.sparseBlockSize);
    uint32_t combineBytes = (constInfo.dSize * sizeof(KV_T) +
                             constInfo.dSizeRope * sizeof(Q_T) +
                             constInfo.dSize / constInfo.tileSize * sizeof(T));
    int64_t keySrcStride = sparseBlockSrcStride * combineBytes;
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
        realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
        realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit) ||
        constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyBNBOffset1, s2IdLimit, runInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyBNBOffset2, s2IdLimit, runInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyBNBOffset1 >= 0) + (keyBNBOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyBNBOffset1 > -1 ? keyBNBOffset1 : keyBNBOffset2;
        if (keyBNBOffset2 > -1 && keyBNBOffset2 < keyBNBOffset1) {
            startGmOffset = keyBNBOffset2;
        }

        // 当前仅支持COMBINE模式
        // if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
            intriParams.blockLen = constInfo.sparseBlockSize * combineBytes;
            uint32_t combineDim = combineBytes / sizeof(KV_T);
            uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = combineDimAlign - combineDim;
            padParams.paddingValue = 0;
            DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) *
                        combineDimAlign], keyGm_[startGmOffset * combineDim], intriParams, padParams);
        // }
        mte2Size += ((keyBNBOffset1 > -1) + (keyBNBOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

#if 0
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::DequantKv(int64_t mergeMte3Idx, int64_t dealRow)
{
    SetFlag<AscendC::HardEvent::MTE2_V>(0);
    WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    LocalTensor<half> kvTensorAsFp16 = tmpBuff1.Get<half>();
    uint64_t mask = ConstInfo::BUFFER_SIZE_BYTE_256B / sizeof(half);
    LocalTensor<KV_T> srcTensor = kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)];
    if (dealRow == 1) {
        Cast(kvTensorAsFp16, srcTensor, RoundMode::CAST_NONE, mask, 4, {1, 1, 8, 4});
    } else {
        uint8_t repeatTimes = static_cast<uint8_t>(dealRow);
        Cast(kvTensorAsFp16, srcTensor, RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, 21}); // 21=(512+64*2+32)/32
        Cast(kvTensorAsFp16[128], srcTensor[128], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, 21});
        Cast(kvTensorAsFp16[256], srcTensor[256], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, 21});
        Cast(kvTensorAsFp16[384], srcTensor[384], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, 21});
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<T> antiQuantScale = tmpBuff2.Get<T>();
    LocalTensor<T> oriQuantScaleTensor = srcTensor[640].template ReinterpretCast<T>();
    if (dealRow == 1) {
        Brcb(antiQuantScale, oriQuantScaleTensor, 1, {1, 4});
    } else {
        DataCopyParams params;
        params.blockCount = dealRow;
        params.blockLen = 1;
        params.srcStride = (constInfo.dSize * sizeof(KV_T) + constInfo.dSizeRope * sizeof(Q_T)) /
            ConstInfo::BUFFER_SIZE_BYTE_32B;
        params.dstStride = 0;
        LocalTensor<T> tmpAntiQuantScale = antiQuantScale[ConstInfo::BUFFER_SIZE_BYTE_1K];
        DataCopy(tmpAntiQuantScale, oriQuantScaleTensor, params);
        PipeBarrier<PIPE_V>();
        Brcb(antiQuantScale, tmpAntiQuantScale, dealRow, {1, 4});
    }
    PipeBarrier<PIPE_V>();
    uint32_t dealLoop = CeilDiv(dealRow, LIMIT_DEAL_ROW);
    uint32_t dealRowFp32 = LIMIT_DEAL_ROW;
    uint32_t element = LIMIT_DEAL_ROW * constInfo.dSize;
    LocalTensor<T> kvTensorAsFp32 = inputBuff2.Get<T>();
    LocalTensor<Q_T> antiKvTensorAsB16 = tmpBuff1.Get<Q_T>();
    for (uint32_t i = 0; i < dealLoop; i++) {
        if (i == dealLoop - 1) {
            dealRowFp32 = dealRow - i * LIMIT_DEAL_ROW;
        }
        Cast(kvTensorAsFp32, kvTensorAsFp16[i * element], RoundMode::CAST_NONE,
            static_cast<uint32_t>(dealRowFp32 * constInfo.dSize));
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < constInfo.tileSize / FP32_REPEAT_ELEMENT_NUM; j++) {
            Mul(kvTensorAsFp32[j * FP32_REPEAT_ELEMENT_NUM], kvTensorAsFp32[j * FP32_REPEAT_ELEMENT_NUM],
                antiQuantScale[i * LIMIT_DEAL_ROW * 32],
                FP32_REPEAT_ELEMENT_NUM, 4 * dealRowFp32, {1, 1, 0, 16, 16, 1});
        }
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<Q_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(antiKvTensorAsB16[i * element], kvTensorAsFp32, RoundMode::CAST_RINT,
                static_cast<uint32_t>(dealRowFp32 * constInfo.dSize));
        } else {
            Cast(antiKvTensorAsB16[i * element], kvTensorAsFp32, RoundMode::CAST_ROUND,
                static_cast<uint32_t>(dealRowFp32 * constInfo.dSize));
        }
        PipeBarrier<PIPE_V>();
    }
}
#else
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_2 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_3 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_4 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ int8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, // output first
                                        __ubuf__ float* ubScaleSrcAddr, uint32_t dealRowCount)
{
    uint32_t combineDim = 512 + 64 * 2 + 4 * 4 + 16; // 32对齐
    MicroAPI::RegTensor<KV_T> vKvData0;
    MicroAPI::RegTensor<KV_T> vKvData1;
    MicroAPI::RegTensor<half> vKvDataHalf0;
    MicroAPI::RegTensor<half> vKvDataHalf1;
    MicroAPI::RegTensor<float> vCastFp32Res0;
    MicroAPI::RegTensor<float> vCastFp32Res1;
    MicroAPI::RegTensor<float> vMulRes0;
    MicroAPI::RegTensor<float> vMulRes1;
    MicroAPI::RegTensor<float> vScale0;
    MicroAPI::RegTensor<float> vScale1;
    MicroAPI::RegTensor<Q_T> vCastRes0;
    MicroAPI::RegTensor<Q_T> vCastRes1;
    MicroAPI::RegTensor<Q_T> vCastResPack0;
    MicroAPI::RegTensor<Q_T> vCastResPack1;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint16_t j = 0; j < (512 / 128); j++) {
        // tilesize is 64, deal 128 b8 kv, deal 2 fp32 scale
        __ubuf__ int8_t* ubSrcTemp = ubSrcAddr + j * 128;
        __ubuf__ float* ubScaleSrcAddrTemp = ubScaleSrcAddr + j * 2;
        __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + j * 128;
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            // load scale
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vKvData0, ubSrcTemp, 64);
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vKvData1, ubSrcTemp, combineDim - 64);

            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                (MicroAPI::RegTensor<int8_t>&)vScale0, ubScaleSrcAddrTemp, 0);
            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                (MicroAPI::RegTensor<int8_t>&)vScale1, ubScaleSrcAddrTemp, combineDim - 0);

            MicroAPI::Cast<half, KV_T, castTraitFp8_1>(vKvDataHalf0, vKvData0, kvTypeMaskAll);
            MicroAPI::Cast<half, KV_T, castTraitFp8_1>(vKvDataHalf1, vKvData1, kvTypeMaskAll);

            MicroAPI::Cast<float, half, castTraitFp8_1>(vCastFp32Res0, vKvDataHalf0, kvTypeMaskAll); // todo mask type
            MicroAPI::Cast<float, half, castTraitFp8_1>(vCastFp32Res1, vKvDataHalf1, kvTypeMaskAll);
#if 1
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes1, vCastFp32Res1, vScale1, fp32MaskAll);

            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vMulRes0, fp32MaskAll);
            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes1, vMulRes1, fp32MaskAll);
#else
            // for debug, skip mul
            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, fp32MaskAll);
            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes1, vCastFp32Res1, fp32MaskAll);
#endif
            MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);
            // todo copy nz
            MicroAPI::StoreAlign<Q_T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, 512, kvRopeTypeMaskAll);
        }
    }
    return;
}
template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T>& outputUb,  LocalTensor<KV_T>& inputUb,
                                   uint32_t dealRowCount) {
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(inputUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(outputUb.GetPhyAddr());
    __ubuf__ float* ubScaleAddr = (__ubuf__ float*)(inputUb[512 + 64 * 2].GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T>(ubSrcAddr, ubDstAddr, ubScaleAddr, dealRowCount);
}
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::DequantKv(int64_t mergeMte3Idx, int64_t dealRow)
{
    LocalTensor<KV_T> srcTensor = kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)];
    LocalTensor<Q_T> antiKvTensorAsB16 = tmpBuff1.Get<Q_T>();
    SetFlag<AscendC::HardEvent::MTE2_V>(0);
    WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    PRINTF("dealrow is %d\n", dealRow);
    DumpTensor(kvMergUb_, 20001, 1024);
    AntiquantVFFp8D448<Q_T, int8_t>(antiKvTensorAsB16, srcTensor, dealRow);
    DumpTensor(antiKvTensorAsB16, 20009, 1024);
}
#endif


#if 1
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2L1(
    int64_t mergeMte3Idx, int64_t nopeGmOffset, int64_t ropeGmOffset, int64_t dealRow)
{
    LocalTensor<KV_T> srcTensor = kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)];
    LocalTensor<Q_T> antiKvTensorAsB16 = tmpBuff1.Get<Q_T>();
    uint64_t mask = ConstInfo::BUFFER_SIZE_BYTE_256B / sizeof(half);

    LocalTensor<Q_T> antiKvTensorAsB16Nz = outputBuff1.Get<Q_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    int dataBlocks = REPEAT_BLOCK_BYTE / BYTE_BLOCK;
    int loops = CeilDiv(dealRow, dataBlocks);
    uint64_t tail = dealRow - (loops - 1) * dataBlocks;
    uint64_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2;
    uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
    uint8_t repeatTimes = static_cast<uint8_t>(constInfo.dSize / blockElementNum);
    for (int i = 0; i < loops; i++) {
        mask = (i == loops - 1) ? tail * blockElementNum : repeatElementNum;
        Copy(antiKvTensorAsB16Nz[i * repeatElementNum], antiKvTensorAsB16[i * dataBlocks * constInfo.dSize], mask,
            repeatTimes, {1, 32, static_cast<uint16_t>(dealRow), 1});
    }
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = constInfo.dSize / blockElementNum;
    dataCopyParams.blockLen = dealRow * blockElementNum * sizeof(Q_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (constInfo.s2BaseSize - dealRow) * blockElementNum * sizeof(Q_T);
    DataCopyPad(kvMergeGm_[nopeGmOffset], antiKvTensorAsB16Nz, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
   
    LocalTensor<Q_T> kRopeUb = srcTensor[512].template ReinterpretCast<Q_T>();
    LocalTensor<Q_T> kRopeUbNz = outputBuff2.Get<Q_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Copy(kRopeUbNz, kRopeUb, constInfo.dSizeRope, static_cast<uint8_t>(dealRow), {static_cast<uint16_t>(dealRow), 1,
        1, 21});
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    dataCopyParams.blockCount = constInfo.dSizeRope / blockElementNum;
    DataCopyPad(kvMergeGm_[ropeGmOffset], kRopeUbNz, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}
#else
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2L1(
    int64_t mergeMte3Idx, int64_t nopeGmOffset, int64_t ropeGmOffset, int64_t dealRow)
{
    uint64_t mask = ConstInfo::BUFFER_SIZE_BYTE_256B / sizeof(half);
    LocalTensor<Q_T> antiKvTensorAsB16 = tmpBuff1.Get<Q_T>();
    LocalTensor<KV_T> srcTensor = kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)];

    LocalTensor<Q_T> antiKvTensorAsB16Nz = outputBuff1.Get<Q_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    int dataBlocks = REPEAT_BLOCK_BYTE / BYTE_BLOCK;
    int loops = CeilDiv(dealRow, dataBlocks);
    uint64_t tail = dealRow - (loops - 1) * dataBlocks;
    uint64_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2;
    uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
    uint8_t repeatTimes = static_cast<uint8_t>(constInfo.dSize / blockElementNum);
    for (int i = 0; i < loops; i++) {
        mask = (i == loops - 1) ? tail * blockElementNum : repeatElementNum;
        Copy(antiKvTensorAsB16Nz[i * repeatElementNum], antiKvTensorAsB16[i * dataBlocks * constInfo.dSize], mask,
            repeatTimes, {1, 32, static_cast<uint16_t>(dealRow), 1});
    }
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = constInfo.dSize / blockElementNum;
    dataCopyParams.blockLen = dealRow * blockElementNum * sizeof(Q_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (constInfo.s2BaseSize - dealRow) * blockElementNum * sizeof(Q_T);
    DataCopyPad(kvMergeGm_[nopeGmOffset], antiKvTensorAsB16Nz, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
   
    LocalTensor<Q_T> kRopeUb = srcTensor[512].template ReinterpretCast<Q_T>();
    LocalTensor<Q_T> kRopeUbNz = outputBuff2.Get<Q_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Copy(kRopeUbNz, kRopeUb, constInfo.dSizeRope, static_cast<uint8_t>(dealRow), {static_cast<uint16_t>(dealRow), 1,
        1, 21});
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    dataCopyParams.blockCount = constInfo.dSizeRope / blockElementNum;
    DataCopyPad(kvMergeGm_[ropeGmOffset], kRopeUbNz, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}
#endif

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                   int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                   const RunInfo &runInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    int32_t dealRow = mte2Size - mte3Size;
    DequantKv(mergeMte3Idx, dealRow);
    uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
    int64_t nopeGmOffset = runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + (s2GmStartOffset +
        mte3Size) * blockElementNum;
    int64_t ropeGmOffset = runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + 512 * 512 + (s2GmStartOffset +
        mte3Size) * blockElementNum;
    CopyOutKvUb2L1(mergeMte3Idx, nopeGmOffset, ropeGmOffset, dealRow);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::MergeOriKv(const RunInfo &runInfo)
{
    // copy gm to ub, deal page attention
    PRINTF("sInner is %d\n", runInfo.s2RealSize);
    int64_t s2V0LoopTimes = runInfo.s2RealSize / 16;
    for (uint32_t i = 0; i < s2V0LoopTimes; i++) {
        int64_t s2ProcessSize = 16;
        int64_t mergeMte3Idx = 0;
        int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : 0;
        int64_t startGmOffset = 0;
        uint32_t combineBytes = (constInfo.dSize * sizeof(KV_T) + constInfo.dSizeRope * sizeof(Q_T) +
            constInfo.dSize / constInfo.tileSize * sizeof(T));
        uint32_t combineDim = combineBytes / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        DataCopyExtParams intriParams;
        intriParams.blockCount = s2ProcessSize;
        intriParams.blockLen = combineBytes;
        intriParams.dstStride = 0;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<KV_T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        if constexpr (isPa) {
            PRINTF("PAGE_ATTENTION=====\n");
            uint64_t blockTableBaseOffset = runInfo.boIdx * constInfo.maxBlockNumPerBatch;
            uint64_t dstOffset = 0;
            uint32_t copyFinishElmenCnt = 0;
            uint32_t curSequence = runInfo.s2BatchOffset;
            while (copyFinishElmenCnt < s2ProcessSize) {
                PRINTF("copyFinishElmenCnt(%d) < s2ProcessSize(%d)\n", copyFinishElmenCnt, s2ProcessSize);
                uint64_t blockIdOffset = curSequence / constInfo.blockSize;
                uint64_t remainElmenCnt = curSequence % constInfo.blockSize;
                uint64_t idInBlockTable = blkTableGm_.GetValue(blockTableBaseOffset + blockIdOffset);
                uint32_t copyElmenCnt = constInfo.blockSize - remainElmenCnt;
                if (copyElmenCnt + copyFinishElmenCnt > s2ProcessSize) {
                    copyElmenCnt = s2ProcessSize - copyFinishElmenCnt;
                }
                uint64_t srcOffset = idInBlockTable * constInfo.blockSize * constInfo.n2Size * combineBytes +
                    remainElmenCnt * constInfo.n2Size * combineBytes + (uint64_t)(runInfo.n2oIdx * combineBytes); // BlockNum, BlockSize, N, D
                intriParams.blockCount = copyElmenCnt; // base s2 size
                DataCopyPad(kvMergUb_[dstOffset], keyGm_[srcOffset], intriParams, padParams);
                dstOffset += copyElmenCnt;
                copyFinishElmenCnt += copyElmenCnt;
                curSequence += copyElmenCnt;
            }
        } else {
            DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)], keyGm_[startGmOffset * combineDim], intriParams, padParams);
        }
        // todo dequant and copy ub to l1
        int32_t dealRow = s2ProcessSize;
        DequantKv(mergeMte3Idx, dealRow);
        mergeMte3Idx++;
        uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
        int64_t nopeGmOffset = runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + (s2GmStartOffset) * blockElementNum;
        int64_t ropeGmOffset = runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + 512 * 512 + (s2GmStartOffset) * blockElementNum;
        CopyOutKvUb2L1(mergeMte3Idx, nopeGmOffset, ropeGmOffset, dealRow);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::MergeKv(const RunInfo &runInfo)
{
    if (0) {
        MergeOriKv(runInfo);
    }
    int64_t s2ProcessSize = runInfo.s2RealSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = 0;

    // if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) { // zhj
    if (constInfo.layoutType == static_cast<uint8_t>(SAS_LAYOUT::TND)) {
        uint64_t actualSeqQPrefixSum = (runInfo.boIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(runInfo.boIdx - 1);
        topkGmBaseOffset += (actualSeqQPrefixSum + runInfo.gS1Idx / constInfo.gSize) * constInfo.n2Size *
                            constInfo.sparseBlockCount + runInfo.n2oIdx * constInfo.sparseBlockCount;
    } else {
        topkGmBaseOffset += runInfo.boIdx * constInfo.s1Size * constInfo.sparseBlockCount +
                            runInfo.gS1Idx / constInfo.gSize * constInfo.sparseBlockCount;
    }
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize: s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit; s2GmOffsetArray += 2 *
        constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, runInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, runInfo);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo);
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > 32) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }

    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        LocalTensor<Q_T> mergeUb = kvMergUb_.template ReinterpretCast<Q_T>();
        Duplicate(mergeUb, static_cast<Q_T>(0.0), constInfo.dSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = constInfo.dSize / blockElementNum;
        dataCopyParams.blockLen = blockElementNum * sizeof(Q_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.s2BaseSize - 1) * blockElementNum * sizeof(Q_T);
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + s2GmOffset * blockElementNum],
                        mergeUb, dataCopyParams);
        }
        dataCopyParams.blockCount = constInfo.dSizeRope / blockElementNum;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + 512 * constInfo.dSize +
                                   s2GmOffset * blockElementNum],
                        mergeUb, dataCopyParams);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        mergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 128 * sizeof(int32_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(kvValidSizeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) + GetSubBlockIdx() * 128],
                v0ValidSizeUb_, dataCopyParams);
    return;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo, 
    ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    // TODO v0尾块填充-inf处理
    if (runInfo.s2LoopCount == 0) {
        if (likely(runInfo.s2RealSize == 128)) {
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, SCFaVectorApi::EQ_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize <= 64){
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, SCFaVectorApi::GT_0_AND_LTE_64_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize < 128){
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, SCFaVectorApi::GT_64_AND_LTE_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (likely(runInfo.s2RealSize == 128)) {
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, SCFaVectorApi::EQ_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, SCFaVectorApi::GT_0_AND_LTE_64_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize < 128){
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, SCFaVectorApi::GT_64_AND_LTE_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();
    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>();

    DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) * (runInfo.s1RealSize - runInfo.halfS1RealSize)], stage1CastTensor,
        {s2BaseSize / 16, (uint16_t)runInfo.halfS1RealSize,
        (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
        (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    // ======================================================
    if (runInfo.s2LoopCount != 0) {
        SCFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo) {
    if (unlikely(runInfo.vec2S1BaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }
    
    // TOTO:为什么是BaseSize
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize; 
    int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, runInfo.vec2S1RealSize);
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(
                vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize);
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut(
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;

    if constexpr (!IsSameType<Q_T, VEC2_RES_T>::value) {
        attenOut.SetAddr(vec2ResUb.address_);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4;
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2S1RealSize;

    int64_t attenOutOffset = constInfo.dSizeV;
    // TODO 确认用模板参数还是constInfo
    // if constexpr (layout == SAS_LAYOUT::TND) {
    if (constInfo.layoutType == static_cast<uint8_t>(SAS_LAYOUT::TND)) {
        attenOutOffset = constInfo.n2GDv;
        if (constInfo.isPfaGS1Merge) {
            attenOutOffset = 0;
            dataCopyParams.blockLen *= constInfo.gSize;
            dataCopyParams.blockCount /= constInfo.gSize;
        } else if (constInfo.isGqa) {
            attenOutOffset = constInfo.dSizeV;
        }
    } else {
        if (constInfo.layoutType == static_cast<uint8_t>(SAS_LAYOUT::BSND)) {
            attenOutOffset = constInfo.n2GDv;
            if (constInfo.isPfaGS1Merge) {
                attenOutOffset = 0;
                dataCopyParams.blockLen *= constInfo.gSize;
                dataCopyParams.blockCount /= constInfo.gSize;
            } else if (constInfo.isGqa) {
                attenOutOffset = constInfo.dSizeV;
            }
        }
    }

    if (constInfo.isPfaGS1Merge && dSizeAligned64 - constInfo.dSizeV != 0 && (constInfo.layoutType == static_cast<uint8_t>(SAS_LAYOUT::BSND) || constInfo.layoutType == static_cast<uint8_t>(SAS_LAYOUT::TND))) {
        for(int64_t i = 0; i < runInfo.vec2S1BaseSize / constInfo.gSize; i++){
            attenOutOffset = i * constInfo.dSizeV * constInfo.gSize * constInfo.n2Size;
            dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
            dataCopyParams.blockCount = constInfo.gSize;
            dataCopyParams.dstStride = 0;
            DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + attenOutOffset],
                attenOut[i * constInfo.gSize * dSizeAligned64], dataCopyParams);
        }
    } else {
        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
            attenOut, dataCopyParams); 
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
{
    auto &initParams = this->tilingData->initOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * initParams.singleCoreSize], singleInitOutputSize, 0.0);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo) 
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        // if (this->tilingData->initOutputParams.needInit == 1) {
        //     InitOutputSingleCore(constInfo);
        // }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
    __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks)
{
    oriKVGm.SetGlobalBuffer((__gm__ KV_T *)(oriKV));
    oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);

    if constexpr (TEMPLATE_MODE != SASTemplateMode::SWA_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
        cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
    }

    if constexpr (TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE) {
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
    }

    if (cuSeqlensQ != nullptr) {
        actualSeqQlenAddr = (__gm__ int64_t *)cuSeqlensQ;
    }
    if (sequsedKv != nullptr) {
        actualSeqKvlenAddr = (__gm__ int64_t *)sequsedKv;
    }
    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ Q_T *)sinks);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    tPipe->InitBuffer(softmaxSumBuf[0], 128); // 64/ 2*sizeof(float) = 128
    tPipe->InitBuffer(softmaxSumBuf[1], 128); 
    tPipe->InitBuffer(softmaxMaxBuf[0], 128); 
    tPipe->InitBuffer(softmaxMaxBuf[1], 128); 
    tPipe->InitBuffer(softmaxExpBuf[0], 128); 
    tPipe->InitBuffer(softmaxExpBuf[1], 128); 
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);

    SoftmaxInitBuffer();

    tPipe->InitBuffer(stage1OutQue[0], 1, 8448); // （32 + 1） * 128 * 2(bf16)
    tPipe->InitBuffer(stage1OutQue[1], 1, 8448);
    tPipe->InitBuffer(stage2OutBuf, 32 * dTemplateAlign64 * sizeof(T)); //s1Base/cv_ratio * 512 * 4(float)

    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();

    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &sparseAttnSharedkvBaseParams = this->tilingData->baseParams;
    sharedParams.bSize = sparseAttnSharedkvBaseParams.batchSize;
    sharedParams.n2Size = 1;
    sharedParams.gSize = sparseAttnSharedkvBaseParams.nNumOfQInOneGroup; 
    sharedParams.s1Size = sparseAttnSharedkvBaseParams.qSeqSize;
    sharedParams.s2Size = sparseAttnSharedkvBaseParams.kvSeqSize;
    sharedParams.dSize = sparseAttnSharedkvBaseParams.actualLenDimsQ;
    sharedParams.dSizeV = sparseAttnSharedkvBaseParams.actualLenDimsKV;
    sharedParams.sparseBlockCount = sparseAttnSharedkvBaseParams.sparseBlockCount;
    sharedParams.sparseBlockSize = sparseAttnSharedkvBaseParams.sparseBlockSize;
    sharedParams.softmaxScale = sparseAttnSharedkvBaseParams.softmaxScale; 
    sharedParams.cmpRatio = sparseAttnSharedkvBaseParams.cmpRatio;
    sharedParams.oriMaskMode = sparseAttnSharedkvBaseParams.oriMaskMode;
    sharedParams.cmpMaskMode = sparseAttnSharedkvBaseParams.cmpMaskMode;
    sharedParams.oriWinLeft = sparseAttnSharedkvBaseParams.oriWinLeft;
    sharedParams.oriWinRight = sparseAttnSharedkvBaseParams.oriWinRight;
    sharedParams.layoutType = sparseAttnSharedkvBaseParams.outputLayout; 
    sharedParams.kvQuantMode = sparseAttnSharedkvBaseParams.kvQuantMode;
    sharedParams.tileSize = sparseAttnSharedkvBaseParams.tileSize;
    sharedParams.ropeHeadDim = sparseAttnSharedkvBaseParams.ropeHeadDim;
    
    // sharedParams.isGqa = sparseAttnSharedkvBaseParams->isGqa; 
    // sharedParams.isPfaGS1Merge = (sparseAttnSharedkvBaseParams->isGqa && sharedParams.s1Size > 1); 
    // sharedParams.actualSeqLengthsSize = sparseAttnSharedkvBaseParams->actualSeqLengthsSize;
    // sharedParams.actualSeqLengthsKVSize = sparseAttnSharedkvBaseParams->actualSeqLengthsKVSize;
    // sharedParams.isActualSeqLengthsNull = sparseAttnSharedkvBaseParams->isActualSeqLengthsNull;
    // sharedParams.isActualSeqLengthsKVNull = sparseAttnSharedkvBaseParams->isActualSeqLengthsKVNull
    // pageAttention
    if constexpr (isPa) {
        // sharedParams.blockSize = sparseAttnSharedkvBaseParams->paBlockSize;
        // sharedParams.paLayoutType = sparseAttnSharedkvBaseParams->paLayoutType;
        // sharedParams.oriMaxBlockNumPerBatch = sparseAttnSharedkvBaseParams->oriMaxBlockNumPerBatch; 
        // sharedParams.cmpMaxBlockNumPerBatch = sparseAttnSharedkvBaseParams->cmpMaxBlockNumPerBatch;
    }
    
    // TODO 补充从metadata获取
    // auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    // sharedParams.s1OuterSize = multiCoreParamsRegbase.s1OuterSize;
    // sharedParams.coreNum = multiCoreParamsRegbase.coreNum;
    sharedParams.s1OuterSize = 64;
    sharedParams.coreNum = 32;
    /* 多核切分偏移计算 */
    // sharedParams.multiCoreInnerOffset = multiCoreParamsRegbase.sparseStartIdx[aicIdx];
    // sharedParams.multiCoreInnerLimit = multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    // sharedParams.bnStartIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx];
    // sharedParams.bnEndIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
    // sharedParams.needInit = this->tilingData->initOutputParams.needInit;

    // sharedParams.multiCoreInnerOffset = multiCoreParamsRegbase.sparseStartIdx[aicIdx];
    // sharedParams.multiCoreInnerLimit = multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    // sharedParams.bnStartIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx];
    // sharedParams.bnEndIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
    // sharedParams.needInit = this->tilingData->initOutputParams.needInit;

    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
            #pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }

}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
    negativeScalar = *((float *)&tmp1);
}

TEMPLATES_DEF
class SCFABlockVecDummy {
public:
    // TODO:需修改为模板参数
    // static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    // static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    // TODO 是否需要补充其他函数
    __aicore__ inline SCFABlockVecDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut,
        ConstInfo &constInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *cmpSparseIndices,
        __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv,
        __gm__ uint8_t *sinks) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseAttnSharedkvTilingData *__restrict tiling,
        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, SasMetaData &metadataLocal) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}
};
}
#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H

