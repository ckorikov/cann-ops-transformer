/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file incre_flash_attention_tiling.h
 * \brief

 */
#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
// #include "tiling/tiling_api.h"
// #include "register/op_def_registry.h"
// #include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling.h"
// #include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling_context.h"

// #ifdef ASCENDC_OP_TEST
// #define IFA_EXTERN_C extern "C"
// #else
// #define IFA_EXTERN_C
// #endif
// namespace optiling {
// #if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionInitOutputParams)
// TILING_DATA_FIELD_DEF(uint32_t, isPerChnOut)
// TILING_DATA_FIELD_DEF(uint32_t, isOutQuantTypeBf16)
// TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize)
// TILING_DATA_FIELD_DEF(uint32_t, singleCoreLseSize)
// TILING_DATA_FIELD_DEF(int64_t, totalOutputSize)
// TILING_DATA_FIELD_DEF(int64_t, totalLseOutputSize)
// TILING_DATA_FIELD_DEF(uint32_t, needInit)
// TILING_DATA_FIELD_DEF(uint32_t, isBSNDOut)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(IncreFlashAttentionInitOutputParamsOp, IncreFlashAttentionInitOutputParams)

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionBaseParams)
// TILING_DATA_FIELD_DEF(uint32_t, batchSize)
// TILING_DATA_FIELD_DEF(uint32_t, seqSize)
// TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
// TILING_DATA_FIELD_DEF(uint32_t, headSize)
// TILING_DATA_FIELD_DEF(uint32_t, headSizeV)
// TILING_DATA_FIELD_DEF(uint32_t, blockSize)
// TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
// TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerSeq)
// TILING_DATA_FIELD_DEF(float, scaleValue)
// TILING_DATA_FIELD_DEF(uint32_t, kvHeadNum)
// TILING_DATA_FIELD_DEF(uint32_t, headNumRatio)
// TILING_DATA_FIELD_DEF(uint32_t, qHeadNum)
// TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
// TILING_DATA_FIELD_DEF(uint32_t, batchContinuousFlag)
// TILING_DATA_FIELD_DEF(uint32_t, pseShiftFlag)
// TILING_DATA_FIELD_DEF(uint32_t, pseShiftB)
// TILING_DATA_FIELD_DEF(uint32_t, pseShiftS)
// TILING_DATA_FIELD_DEF(uint32_t, pseShiftS0)
// TILING_DATA_FIELD_DEF(uint32_t, selectWithByteMaskTmpMinSize)
// TILING_DATA_FIELD_DEF(uint32_t, actualLenQDims)
// TILING_DATA_FIELD_DEF(uint32_t, actualLenDims)
// TILING_DATA_FIELD_DEF(uint32_t, qPaddingFlag)
// TILING_DATA_FIELD_DEF(uint32_t, kvPaddingFlag)
// TILING_DATA_FIELD_DEF(uint32_t, msdIterNum)
// TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
// TILING_DATA_FIELD_DEF(uint32_t, antiquantPerTensorFlag)
// TILING_DATA_FIELD_DEF(uint32_t, antiquantPerHeadFlag)
// TILING_DATA_FIELD_DEF(uint32_t, antiquantParamsInPagedAttentionFlag)
// TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
// TILING_DATA_FIELD_DEF(uint32_t, attenMaskBatch)
// TILING_DATA_FIELD_DEF(uint32_t, attenMaskQSize)
// TILING_DATA_FIELD_DEF(uint32_t, attenMaskSize)
// TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
// TILING_DATA_FIELD_DEF(uint32_t, totalBlockNum)
// TILING_DATA_FIELD_DEF(uint32_t, paKvShapeType)
// TILING_DATA_FIELD_DEF(uint32_t, antiqSeqSize)
// TILING_DATA_FIELD_DEF(int32_t, preToken)
// TILING_DATA_FIELD_DEF(int32_t, nextToken)
// TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid)
// TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
// TILING_DATA_FIELD_DEF(uint32_t, slidingFlag)
// TILING_DATA_FIELD_DEF(int64_t, windowSize)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(IncreFlashAttentionBaseParamsOp, IncreFlashAttentionBaseParams)

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionCoreParams)
// TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreSidxEnd); // 50:MAX_CORE_NUM of 910b coreSidxEnd数组首地址要保证8字节对齐
// TILING_DATA_FIELD_DEF_ARR(uint32_t, 66, coreSidxEndRegbase); // 66:MAX_CORE_NUM of 910_95
// TILING_DATA_FIELD_DEF_ARR(uint32_t, 66, coreSposStartRegbase); // 66:MAX_CORE_NUM of 910_95
// END_TILING_DATA_DEF;
// REGISTER_TILING_DATA_CLASS(IncreFlashAttentionCoreParamsOp, IncreFlashAttentionCoreParams);

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionSplitCoreParams)
// TILING_DATA_FIELD_DEF(uint32_t, headSplit)
// TILING_DATA_FIELD_DEF(uint32_t, maskHeadStride)
// TILING_DATA_FIELD_DEF(uint32_t, maskBatchStride)
// TILING_DATA_FIELD_DEF(uint32_t, qTokens)
// TILING_DATA_FIELD_DEF(uint32_t, isTriu)
// TILING_DATA_FIELD_DEF(uint32_t, maxSeqlen)
// TILING_DATA_FIELD_DEF(uint32_t, totalQBlockNum)
// TILING_DATA_FIELD_DEF(uint32_t, seqStepQ)
// TILING_DATA_FIELD_DEF(uint32_t, seqStepKv)
// TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, startBlk); 
// TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, endBlk); 
// TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, startBatch); 
// TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, endBatch);
// END_TILING_DATA_DEF;
// REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSplitCoreParamsOp, IncreFlashAttentionSplitCoreParams);

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionSingleCoreParams)
// TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopTimes);
// TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
// TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSizeTail);
// TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
// TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
// TILING_DATA_FIELD_DEF(uint32_t, blockSplitBn2Range);
// TILING_DATA_FIELD_DEF(uint32_t, tailSplitedBatchRange);
// TILING_DATA_FIELD_DEF(uint32_t, groupSplitSize);
// TILING_DATA_FIELD_DEF(uint32_t, s1SplitSize);
// END_TILING_DATA_DEF;
// REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSingleCoreParamsOp, IncreFlashAttentionSingleCoreParams)

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionSingleCoreTensorSize)
// TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
// TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
// END_TILING_DATA_DEF;
// REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSingleCoreTensorSizeOp, IncreFlashAttentionSingleCoreTensorSize)

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionSplitKVParams)
// TILING_DATA_FIELD_DEF(uint32_t, s2)
// TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopSize)
// TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)
// TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize)
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSplitKVParamsOp, IncreFlashAttentionSplitKVParams)

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingData)
// TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
// TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionBaseParams, baseParams);
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSplitKVParams, splitKVParams);
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionCoreParams, increFlashAttentionCoreParams);
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSingleCoreParams, increFlashAttentionSingleCoreParams);
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSingleCoreTensorSize, increFlashAttentionSingleCoreTensorSize);
// TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionInitOutputParams, outputParams);
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(IncreFlashAttentionTilingDataOp, IncreFlashAttentionTilingData)

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingDataPrefix)
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionTilingData, base);
// TILING_DATA_FIELD_DEF(uint64_t, prefixAttenOutOffset); // 临时输出偏移
// TILING_DATA_FIELD_DEF(uint64_t, userPromptAttenOutOffset);
// TILING_DATA_FIELD_DEF(uint64_t, tmpLseOffset);
// TILING_DATA_FIELD_DEF(uint64_t, prefixLen); // prefix 长度
// TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum); // combine 分核参数，参考普通bn分核流程，总数不超过blockdim
// TILING_DATA_FIELD_DEF(uint32_t, blockSplitBn2Range);
// TILING_DATA_FIELD_DEF(uint32_t, tailSplitedBatchRange);
// TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
// TILING_DATA_FIELD_DEF(uint32_t, batchSizeQ);
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(IncreFlashAttentionTilingDataPrefixOp, IncreFlashAttentionTilingDataPrefix)

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingDataV2)
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionTilingData, tilingBase);
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionTilingDataPrefix, tilingPrefix);
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(IncreFlashAttention, IncreFlashAttentionTilingDataV2)

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingAtbDataV2)
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionBaseParams, tilingBase);
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSplitCoreParams, tilingPerCore);
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(IncreFlashAttention_30000000000200000, IncreFlashAttentionTilingAtbDataV2)
// REGISTER_TILING_DATA_CLASS(IncreFlashAttention_30000000000200001, IncreFlashAttentionTilingAtbDataV2)
// REGISTER_TILING_DATA_CLASS(IncreFlashAttention_30000000000200302, IncreFlashAttentionTilingAtbDataV2)
// REGISTER_TILING_DATA_CLASS(IncreFlashAttention_30000000000222322, IncreFlashAttentionTilingAtbDataV2)

// BEGIN_TILING_DATA_DEF(IncreFlashAttentionEmptyInputTilingData)
// TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionInitOutputParams, outputParams);
// END_TILING_DATA_DEF
// REGISTER_TILING_DATA_CLASS(IncreFlashAttention_13, IncreFlashAttentionEmptyInputTilingData)
// REGISTER_TILING_DATA_CLASS(IncreFlashAttention_14, IncreFlashAttentionEmptyInputTilingData)
// REGISTER_TILING_DATA_CLASS(IncreFlashAttention_27, IncreFlashAttentionEmptyInputTilingData)
// REGISTER_TILING_DATA_CLASS(IncreFlashAttention_30, IncreFlashAttentionEmptyInputTilingData)

// REGISTER_TILING_DATA_CLASS(IncreFlashAttention_1000000000000000090, FlashAttentionScoreSimplifiedTilingData)
// REGISTER_TILING_DATA_CLASS(IncreFlashAttention_1000000000000000020, PromptFlashAttentionTilingData)

// #else

class IncreFlashAttentionBaseParams {
public:
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t qSeqSize;
    uint32_t headSize;
    uint32_t headSizeV;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t maxBlockNumPerSeq;
    float scaleValue;
    uint32_t kvHeadNum;
    uint32_t headNumRatio;
    uint32_t qHeadNum;
    uint32_t nNumOfQInOneGroup;
    uint32_t batchContinuousFlag;
    uint32_t pseShiftFlag;
    uint32_t pseShiftB;
    uint32_t pseShiftS;
    uint32_t pseShiftS0;
    uint32_t selectWithByteMaskTmpMinSize;
    uint32_t actualLenQDims;
    uint32_t actualLenDims;
    uint32_t qPaddingFlag;
    uint32_t kvPaddingFlag;
    uint32_t msdIterNum;
    uint32_t l2CacheOffFlag;
    uint32_t antiquantPerTensorFlag;
    uint32_t antiquantPerHeadFlag;
    uint32_t antiquantParamsInPagedAttentionFlag;
    uint32_t attenMaskFlag;
    uint32_t attenMaskBatch;
    uint32_t attenMaskQSize;
    uint32_t attenMaskSize;
    uint32_t softmaxLseFlag;
    uint32_t totalBlockNum;
    uint32_t paKvShapeType;
    uint32_t antiqSeqSize;
    int32_t preToken;
    int32_t nextToken;
    uint32_t isRowInvalid;
    uint32_t sparseMode;
    uint32_t slidingFlag;
    int64_t windowSize;

    uint32_t get_batchSize() { return batchSize; }
    uint32_t get_seqSize() { return seqSize; }
    uint32_t get_qSeqSize() { return qSeqSize; }
    uint32_t get_headSize() { return headSize; }
    uint32_t get_headSizeV() { return headSizeV; }
    uint32_t get_blockSize() { return blockSize; }
    uint32_t get_maxBlockNumPerBatch() { return maxBlockNumPerBatch; }
    uint32_t get_maxBlockNumPerSeq() { return maxBlockNumPerSeq; }
    float get_scaleValue() { return scaleValue; }
    uint32_t get_kvHeadNum() { return kvHeadNum; }
    uint32_t get_headNumRatio() { return headNumRatio; }
    uint32_t get_qHeadNum() { return qHeadNum; }
    uint32_t get_nNumOfQInOneGroup() { return nNumOfQInOneGroup; }
    uint32_t get_batchContinuousFlag() { return batchContinuousFlag; }
    uint32_t get_pseShiftFlag() { return pseShiftFlag; }
    uint32_t get_pseShiftB() { return pseShiftB; }
    uint32_t get_pseShiftS() { return pseShiftS; }
    uint32_t get_pseShiftS0() { return pseShiftS0; }
    uint32_t get_selectWithByteMaskTmpMinSize() { return selectWithByteMaskTmpMinSize; }
    uint32_t get_actualLenQDims() { return actualLenQDims; }
    uint32_t get_actualLenDims() { return actualLenDims; }
    uint32_t get_qPaddingFlag() { return qPaddingFlag; }
    uint32_t get_kvPaddingFlag() { return kvPaddingFlag; }
    uint32_t get_msdIterNum() { return msdIterNum; }
    uint32_t get_l2CacheOffFlag() { return l2CacheOffFlag; }
    uint32_t get_antiquantPerTensorFlag() { return antiquantPerTensorFlag; }
    uint32_t get_antiquantPerHeadFlag() { return antiquantPerHeadFlag; }
    uint32_t get_antiquantParamsInPagedAttentionFlag() { return antiquantParamsInPagedAttentionFlag; }
    uint32_t get_attenMaskFlag() { return attenMaskFlag; }
    uint32_t get_attenMaskBatch() { return attenMaskBatch; }
    uint32_t get_attenMaskQSize() { return attenMaskQSize; }
    uint32_t get_attenMaskSize() { return attenMaskSize; }
    uint32_t get_softmaxLseFlag() { return softmaxLseFlag; }
    uint32_t get_totalBlockNum() { return totalBlockNum; }
    uint32_t get_paKvShapeType() { return paKvShapeType; }
    uint32_t get_antiqSeqSize() { return antiqSeqSize; }
    int32_t get_preToken() { return preToken; }
    int32_t get_nextToken() { return nextToken; }
    uint32_t get_isRowInvalid() { return isRowInvalid; }
    uint32_t get_sparseMode() { return sparseMode; }
    uint32_t get_slidingFlag() { return slidingFlag; }
    int64_t get_windowSize() { return windowSize; }

    void set_batchSize(uint32_t batchSize) { this->batchSize = batchSize; }
    void set_seqSize(uint32_t seqSize) { this->seqSize = seqSize; }
    void set_qSeqSize(uint32_t qSeqSize) { this->qSeqSize = qSeqSize; }
    void set_headSize(uint32_t headSize) { this->headSize = headSize; }
    void set_headSizeV(uint32_t headSizeV) { this->headSizeV = headSizeV; }
    void set_blockSize(uint32_t blockSize) { this->blockSize = blockSize; }
    void set_maxBlockNumPerBatch(uint32_t maxBlockNumPerBatch) { this->maxBlockNumPerBatch = maxBlockNumPerBatch; }
    void set_maxBlockNumPerSeq(uint32_t maxBlockNumPerSeq) { this->maxBlockNumPerSeq = maxBlockNumPerSeq; }
    void set_scaleValue(float scaleValue) { this->scaleValue = scaleValue; }
    void set_kvHeadNum(uint32_t kvHeadNum) { this->kvHeadNum = kvHeadNum; }
    void set_headNumRatio(uint32_t headNumRatio) { this->headNumRatio = headNumRatio; }
    void set_qHeadNum(uint32_t qHeadNum) { this->qHeadNum = qHeadNum; }
    void set_nNumOfQInOneGroup(uint32_t nNumOfQInOneGroup) { this->nNumOfQInOneGroup = nNumOfQInOneGroup; }
    void set_batchContinuousFlag(uint32_t batchContinuousFlag) { this->batchContinuousFlag = batchContinuousFlag; }
    void set_pseShiftFlag(uint32_t pseShiftFlag) { this->pseShiftFlag = pseShiftFlag; }
    void set_pseShiftB(uint32_t pseShiftB) { this->pseShiftB = pseShiftB; }
    void set_pseShiftS(uint32_t pseShiftS) { this->batchSize = pseShiftS; }
    void set_pseShiftS0(uint32_t pseShiftS0) { this->pseShiftS0 = pseShiftS0; }
    void set_selectWithByteMaskTmpMinSize(uint32_t selectWithByteMaskTmpMinSize) { this->selectWithByteMaskTmpMinSize = selectWithByteMaskTmpMinSize; }
    void set_actualLenQDims(uint32_t actualLenQDims) { this->actualLenQDims = actualLenQDims; }
    void set_actualLenDims(uint32_t actualLenDims) { this->actualLenDims = actualLenDims; }
    void set_qPaddingFlag(uint32_t qPaddingFlag) { this->qPaddingFlag = qPaddingFlag; }
    void set_kvPaddingFlag(uint32_t kvPaddingFlag) { this->kvPaddingFlag = kvPaddingFlag; }
    void set_msdIterNum(uint32_t msdIterNum) { this->msdIterNum = msdIterNum; }
    void set_l2CacheOffFlag(uint32_t l2CacheOffFlag) { this->l2CacheOffFlag = l2CacheOffFlag; }
    void set_antiquantPerTensorFlag(uint32_t antiquantPerTensorFlag) { this->antiquantPerTensorFlag = antiquantPerTensorFlag; }
    void set_antiquantPerHeadFlag(uint32_t antiquantPerHeadFlag) { this->antiquantPerHeadFlag = antiquantPerHeadFlag; }
    void set_antiquantParamsInPagedAttentionFlag(uint32_t antiquantParamsInPagedAttentionFlag) { this->antiquantParamsInPagedAttentionFlag = antiquantParamsInPagedAttentionFlag; }
    void set_attenMaskFlag(uint32_t attenMaskFlag) { this->attenMaskFlag = attenMaskFlag; }
    void set_attenMaskBatch(uint32_t attenMaskBatch) { this->attenMaskBatch = attenMaskBatch; }
    void set_attenMaskQSize(uint32_t attenMaskQSize) { this->attenMaskQSize = attenMaskQSize; }
    void set_attenMaskSize(uint32_t attenMaskSize) { this->attenMaskSize = attenMaskSize; }
    void set_softmaxLseFlag(uint32_t softmaxLseFlag) { this->softmaxLseFlag = softmaxLseFlag; }
    void set_totalBlockNum(uint32_t totalBlockNum) { this->totalBlockNum = totalBlockNum; }
    void set_paKvShapeType(uint32_t paKvShapeType) { this->paKvShapeType = paKvShapeType; }
    void set_antiqSeqSize(uint32_t antiqSeqSize) { this->antiqSeqSize = antiqSeqSize; }
    void set_preToken(int32_t preToken) { this->preToken = preToken; }
    void set_nextToken(int32_t nextToken) { this->nextToken = nextToken; }
    void set_isRowInvalid(uint32_t isRowInvalid) { this->isRowInvalid = isRowInvalid; }
    void set_sparseMode(uint32_t sparseMode) { this->sparseMode = sparseMode; }
    void set_slidingFlag(uint32_t slidingFlag) { this->slidingFlag = slidingFlag; }
    void set_windowSize(uint64_t windowSize) { this->windowSize = windowSize; }
};

class IncreFlashAttentionCoreParams {
public:
    uint32_t coreSidxEnd[50]; // 50:MAX_CORE_NUM of 910b coreSidxEnd数组首地址要保证8字节对齐
    uint32_t coreSidxEndRegbase[66]; // 66:MAX_CORE_NUM of 910_95
    uint32_t coreSposStartRegbase[66]; // 66:MAX_CORE_NUM of 910_95

    uint32_t* get_coreSidxEnd() { return coreSidxEnd; }
    uint32_t* get_coreSidxEndRegbase() { return coreSidxEndRegbase; }
    uint32_t* get_coreSposStartRegbase() { return coreSposStartRegbase; }
};

class IncreFlashAttentionSplitCoreParams {
public:
    uint32_t headSplit;
    uint32_t maskHeadStride;
    uint32_t maskBatchStride;
    uint32_t qTokens;
    uint32_t isTriu;
    uint32_t maxSeqlen;
    uint32_t totalQBlockNum;
    uint32_t seqStepQ;
    uint32_t seqStepKv;
    uint32_t startBlk[50];
    uint32_t endBlk[50];
    uint32_t startBatch[50];
    uint32_t endBatch[50];

    uint32_t get_headSplit() { return headSplit; }
    uint32_t get_maskHeadStride() { return maskHeadStride; }
    uint32_t get_maskBatchStride() { return maskBatchStride; }
    uint32_t get_qTokens() { return qTokens; }
    uint32_t get_isTriu() { return isTriu; }
    uint32_t get_maxSeqlen() { return maxSeqlen; }
    uint32_t get_totalQBlockNum() { return totalQBlockNum; }
    uint32_t get_seqStepQ() { return seqStepQ; }
    uint32_t get_seqStepKv() { return seqStepKv; }
    const uint32_t* get_startBlk() { return startBlk; }
    const uint32_t* get_endBlk() { return endBlk; }
    const uint32_t* get_startBatch() { return startBatch; }
    const uint32_t* get_endBatch() { return endBatch; }

    void set_headSplit(uint32_t headSplit) { this->headSplit = headSplit; }
    void set_maskHeadStride(uint32_t maskHeadStride) { this->maskHeadStride = maskHeadStride; }
    void set_maskBatchStride(uint32_t maskBatchStride) { this->maskBatchStride = maskBatchStride; }
    void set_qTokens(uint32_t qTokens) { this->qTokens = qTokens; }
    void set_isTriu(uint32_t isTriu) { this->isTriu = isTriu; }
    void set_maxSeqlen(uint32_t maxSeqlen) { this->maxSeqlen = maxSeqlen; }
    void set_totalQBlockNum(uint32_t totalQBlockNum) { this->totalQBlockNum = totalQBlockNum; }
    void set_seqStepQ(uint32_t seqStepQ) { this->seqStepQ = seqStepQ; }
    void set_seqStepKv(uint32_t seqStepKv) { this->seqStepKv = seqStepKv; }
    void set_startBlk(const uint32_t* startBlk) { 
        for (int i = 0; i < 50; i++) {
            this->startBlk[i] = startBlk[i];
        }
    }
    void set_endBlk(const uint32_t* endBlk) { 
        for (int i = 0; i < 50; i++) {
            this->endBlk[i] = endBlk[i];
        }
    }  
    void set_startBatch(const uint32_t* startBatch) {
        for (int i = 0; i < 50; i++) {
            this->startBatch[i] = startBatch[i];
        }
    }
    void set_endBatch(const uint32_t* endBatch) {
        for (int i = 0; i < 50; i++) {
            this->endBatch[i] = endBatch[i];
        }
    }
};

class IncreFlashAttentionSingleCoreParams{
public:
    uint32_t sInnerLoopTimes;
    uint32_t singleProcessSInnerSize;
    uint32_t singleProcessSInnerSizeTail;
    uint32_t usedCoreNum;
    uint32_t formerCoreNum;
    uint32_t blockSplitBn2Range;
    uint32_t tailSplitedBatchRange;
    uint32_t groupSplitSize;
    uint32_t s1SplitSize;

    uint32_t get_sInnerLoopTimes() { return sInnerLoopTimes; }
    uint32_t get_singleProcessSInnerSize() { return singleProcessSInnerSize; }
    uint32_t get_singleProcessSInnerSizeTail() { return singleProcessSInnerSizeTail; }
    uint32_t get_usedCoreNum() { return usedCoreNum; }
    uint32_t get_formerCoreNum() { return formerCoreNum; }
    uint32_t get_blockSplitBn2Range() { return blockSplitBn2Range; }
    uint32_t get_tailSplitedBatchRange() { return tailSplitedBatchRange; }
    uint32_t get_groupSplitSize() { return groupSplitSize; }
    uint32_t get_s1SplitSize() { return s1SplitSize; }

    void set_sInnerLoopTimes(uint32_t sInnerLoopTimes) { this->sInnerLoopTimes = sInnerLoopTimes; }
    void set_singleProcessSInnerSize(uint32_t singleProcessSInnerSize) { this->singleProcessSInnerSize = singleProcessSInnerSize; }
    void set_singleProcessSInnerSizeTail(uint32_t singleProcessSInnerSizeTail) { this->singleProcessSInnerSizeTail = singleProcessSInnerSizeTail; }
    void set_usedCoreNum(uint32_t usedCoreNum) { this->usedCoreNum = usedCoreNum; }
    void set_formerCoreNum(uint32_t formerCoreNum) { this->formerCoreNum = formerCoreNum; }
    void set_blockSplitBn2Range(uint32_t blockSplitBn2Range) { this->blockSplitBn2Range = blockSplitBn2Range; }
    void set_tailSplitedBatchRange(uint32_t tailSplitedBatchRange) { this->tailSplitedBatchRange = tailSplitedBatchRange; }
    void set_groupSplitSize(uint32_t groupSplitSize) { this->groupSplitSize = groupSplitSize; }
    void set_s1SplitSize(uint32_t s1SplitSize) { this->s1SplitSize = s1SplitSize; }
};

class IncreFlashAttentionSingleCoreTensorSize {
public:
    uint32_t mmResUbSize; 
    uint32_t bmm2ResUbSize;

    uint32_t get_mmResUbSize() { return mmResUbSize; }
    uint32_t get_bmm2ResUbSize() { return bmm2ResUbSize; }

    void set_mmResUbSize(uint32_t mmResUbSize) { this->mmResUbSize = mmResUbSize; }
    void set_bmm2ResUbSize(uint32_t bmm2ResUbSize) { this->bmm2ResUbSize = bmm2ResUbSize; }
};

class IncreFlashAttentionInitOutputParams {
public:
    uint32_t isPerChnOut;
    uint32_t isOutQuantTypeBf16;
    uint32_t singleCoreSize;
    uint32_t singleCoreLseSize;
    int64_t totalOutputSize;
    int64_t totalLseOutputSize;
    uint32_t needInit;
    uint32_t isBSNDOut;

    uint32_t get_isPerChnOut() { return isPerChnOut; }
    uint32_t get_isOutQuantTypeBf16() { return isOutQuantTypeBf16; }
    uint32_t get_singleCoreSize() { return singleCoreSize; }
    uint32_t get_singleCoreLseSize() { return singleCoreLseSize; }
    int64_t get_totalOutputSize() { return totalOutputSize; }
    int64_t get_totalLseOutputSize() { return totalLseOutputSize; }
    uint32_t get_needInit() { return needInit; }
    uint32_t get_isBSNDOut() { return isBSNDOut; }

    void set_isPerChnOut(uint32_t isPerChnOut) { this->isPerChnOut = isPerChnOut; }
    void set_isOutQuantTypeBf16(uint32_t isOutQuantTypeBf16) { this->isOutQuantTypeBf16 = isOutQuantTypeBf16; }
    void set_singleCoreSize(uint32_t singleCoreSize) { this->singleCoreSize = singleCoreSize; }
    void set_singleCoreLseSize(uint32_t singleCoreLseSize) { this->singleCoreLseSize = singleCoreLseSize; }
    void set_totalOutputSize(int64_t totalOutputSize) { this->totalOutputSize = totalOutputSize; }
    void set_totalLseOutputSize(int64_t totalLseOutputSize) { this->totalLseOutputSize = totalLseOutputSize; }
    void set_needInit(uint32_t needInit) { this->needInit = needInit; }
    void set_isBSNDOut(uint32_t isBSNDOut) { this->isBSNDOut = isBSNDOut; }
};

class IncreFlashAttentionSplitKVParams {
public:
    uint32_t s2; 
    uint32_t sInnerLoopSize;
    uint32_t accumOutSize; 
    uint32_t logSumExpSize;

    uint32_t get_s2() { return s2; }
    uint32_t get_sInnerLoopSize() { return sInnerLoopSize; }
    uint32_t get_accumOutSize() { return accumOutSize; }
    uint32_t get_logSumExpSize() { return logSumExpSize; }
    
    void set_s2(uint32_t s2) { this->s2 = s2; }
    void set_sInnerLoopSize(uint32_t sInnerLoopSize) { this->sInnerLoopSize = sInnerLoopSize; }
    void set_accumOutSize(uint32_t accumOutSize) { this->accumOutSize = accumOutSize; }
    void set_logSumExpSize(uint32_t logSumExpSize) { this->logSumExpSize = logSumExpSize; }
};

class IncreFlashAttentionTilingData {
public:
    TCubeTiling bmm1TilingData;
    TCubeTiling bmm2TilingData;
    IncreFlashAttentionBaseParams baseParams;
    IncreFlashAttentionSplitKVParams splitKVParams;
    IncreFlashAttentionCoreParams increFlashAttentionCoreParams;
    IncreFlashAttentionSingleCoreParams increFlashAttentionSingleCoreParams;
    IncreFlashAttentionSingleCoreTensorSize increFlashAttentionSingleCoreTensorSize;
    SoftMaxTiling softmaxFlashTilingData;
    IncreFlashAttentionInitOutputParams outputParams;
};

class IncreFlashAttentionTilingDataPrefix {
public:
    IncreFlashAttentionTilingData base;
    uint64_t prefixAttenOutOffset;
    uint64_t userPromptAttenOutOffset;
    uint64_t tmpLseOffset;
    uint64_t prefixLen;
    uint32_t formerCoreNum;
    uint32_t blockSplitBn2Range;
    uint32_t tailSplitedBatchRange;
    uint32_t usedCoreNum;
    uint32_t batchSizeQ;

    uint64_t get_prefixAttenOutOffset() { return prefixAttenOutOffset; }
    uint64_t get_userPromptAttenOutOffset() { return userPromptAttenOutOffset; }
    uint64_t get_tmpLseOffset() { return tmpLseOffset; }
    uint64_t get_prefixLen() { return prefixLen; }
    uint32_t get_formerCoreNum() { return formerCoreNum; }
    uint32_t get_blockSplitBn2Range() { return blockSplitBn2Range; }
    uint32_t get_tailSplitedBatchRange() { return tailSplitedBatchRange; }
    uint32_t get_usedCoreNum() { return usedCoreNum; }
    uint32_t get_batchSizeQ() { return batchSizeQ; }

    void set_prefixAttenOutOffset(uint64_t prefixAttenOutOffset) { this->prefixAttenOutOffset = prefixAttenOutOffset; }
    void set_userPromptAttenOutOffset(uint64_t userPromptAttenOutOffset) { this->userPromptAttenOutOffset = userPromptAttenOutOffset; }
    void set_tmpLseOffset(uint64_t tmpLseOffset) { this->tmpLseOffset = tmpLseOffset; }
    void set_prefixLen(uint64_t prefixLen) { this->prefixLen = prefixLen; }
    void set_formerCoreNum(uint32_t formerCoreNum) { this->formerCoreNum = formerCoreNum; }
    void set_blockSplitBn2Range(uint32_t blockSplitBn2Range) { this->blockSplitBn2Range = blockSplitBn2Range; }
    void set_tailSplitedBatchRange(uint32_t tailSplitedBatchRange) { this->tailSplitedBatchRange = tailSplitedBatchRange; }
    void set_usedCoreNum(uint32_t usedCoreNum) { this->usedCoreNum = usedCoreNum; }
    void set_batchSizeQ(uint32_t batchSizeQ) { this->batchSizeQ = batchSizeQ; }
};

class IncreFlashAttentionTilingDataV2 {
public:
    IncreFlashAttentionTilingData tilingBase;
    IncreFlashAttentionTilingDataPrefix tilingPrefix;
};

class IncreFlashAttentionTilingAtbDataV2 {
public:
    IncreFlashAttentionBaseParams tilingBase;
    IncreFlashAttentionSplitCoreParams tilingPerCore;
};

class IncreFlashAttentionEmptyInputTilingData {
public:
    IncreFlashAttentionInitOutputParams outputParams;
};

const uint32_t MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

class IncreFlashAttentionBaseParamsMla{
public:
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t qSeqSize;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    float scaleValue;
    uint32_t nNumOfQInOneGroup;
    uint32_t actualLenQDims;
    uint32_t actualLenDims;
    uint32_t antiquantMode;
    uint32_t attenMaskFlag;
    uint32_t attenMaskSize;
    uint32_t outputLayout;

    uint32_t get_batchSize() { return batchSize; }
    uint32_t get_seqSize() { return seqSize; }
    uint32_t get_qSeqSize() { return qSeqSize; }
    uint32_t get_blockSize() { return blockSize; }
    uint32_t get_maxBlockNumPerBatch() { return maxBlockNumPerBatch; }
    float get_scaleValue() { return scaleValue; }
    uint32_t get_nNumOfQInOneGroup() { return nNumOfQInOneGroup; }
    uint32_t get_actualLenQDims() { return actualLenQDims; }
    uint32_t get_actualLenDims() { return actualLenDims; }
    uint32_t get_antiquantMode() { return antiquantMode; }
    uint32_t get_attenMaskFlag() { return attenMaskFlag; }
    uint32_t get_attenMaskSize() { return attenMaskSize; }
    uint32_t get_outputLayout() { return outputLayout; }

    void set_batchSize(uint32_t batchSize) { this->batchSize = batchSize; }
    void set_seqSize(uint32_t seqSize) { this->seqSize = seqSize; }
    void set_qSeqSize(uint32_t qSeqSize) { this->qSeqSize = qSeqSize; }
    void set_blockSize(uint32_t blockSize) { this->blockSize = blockSize; }
    void set_maxBlockNumPerBatch(uint32_t maxBlockNumPerBatch) { this->maxBlockNumPerBatch = maxBlockNumPerBatch; }
    void set_scaleValue(float scaleValue) { this->scaleValue = scaleValue; }
    void set_nNumOfQInOneGroup(uint32_t nNumOfQInOneGroup) { this->nNumOfQInOneGroup = nNumOfQInOneGroup; }
    void set_actualLenQDims(uint32_t actualLenQDims) { this->actualLenQDims = actualLenQDims; }
    void set_actualLenDims(uint32_t actualLenDims) { this->actualLenDims = actualLenDims; }
    void set_antiquantMode(uint32_t antiquantMode) { this->antiquantMode = antiquantMode; }
    void set_attenMaskFlag(uint32_t attenMaskFlag) { this->attenMaskFlag = attenMaskFlag; }
    void set_attenMaskSize(uint32_t attenMaskSize) { this->attenMaskSize = attenMaskSize; }
    void set_outputLayout(uint32_t outputLayout) { this->outputLayout = outputLayout; }
};

class IncreFlashAttentionCoreParamsMla {
public:
    uint32_t coreBEnd[MAX_AIC_CORE_NUM]; 
    uint32_t coreSidxEnd[MAX_AIC_CORE_NUM];
    uint32_t coreS1OuterEnd[MAX_AIC_CORE_NUM]; 
    uint32_t coreS2End[MAX_AIC_CORE_NUM]; 

    uint32_t* get_coreBEnd() { return coreBEnd; }
    uint32_t* get_coreSidxEnd() { return coreSidxEnd; }
    uint32_t* get_coreS1OuterEnd() { return coreS1OuterEnd; }
    uint32_t* get_coreS2End() { return coreS2End; }
};

class IncreFlashAttentionSingleCoreParamsMla {
public:
    uint32_t singleProcessSInnerSize; 
    uint32_t usedCoreNum;
    uint32_t groupSplitSize; 
    uint32_t s1SplitSize; 

    uint32_t get_singleProcessSInnerSize() { return singleProcessSInnerSize; }
    uint32_t get_usedCoreNum() { return usedCoreNum; }
    uint32_t get_groupSplitSize() { return groupSplitSize; }
    uint32_t get_s1SplitSize() { return s1SplitSize; }

    void set_singleProcessSInnerSize(uint32_t singleProcessSInnerSize) { this->singleProcessSInnerSize = singleProcessSInnerSize; }
    void set_usedCoreNum(uint32_t usedCoreNum) { this->usedCoreNum = usedCoreNum; }
    void set_groupSplitSize(uint32_t groupSplitSize) { this->groupSplitSize = groupSplitSize; }
    void set_s1SplitSize(uint32_t s1SplitSize) { this->s1SplitSize = s1SplitSize; }
};

class IncreFlashAttentionSingleCoreTensorSizeMla {
public:
    uint32_t mmResUbSize; 
    uint32_t bmm2ResUbSize;

    uint32_t get_mmResUbSize() { return mmResUbSize; }
    uint32_t get_bmm2ResUbSize() { return bmm2ResUbSize; }

    void set_mmResUbSize(uint32_t mmResUbSize) { this->mmResUbSize = mmResUbSize; }
    void set_bmm2ResUbSize(uint32_t bmm2ResUbSize) { this->bmm2ResUbSize = bmm2ResUbSize; }
};

class IncreFlashAttentionSplitKVParamsMla {
public:
    uint32_t s2; 
    uint32_t sInnerLoopSize;
    uint32_t accumOutSize; 
    uint32_t logSumExpSize;

    uint32_t get_s2() { return s2; }
    uint32_t get_sInnerLoopSize() { return sInnerLoopSize; }
    uint32_t get_accumOutSize() { return accumOutSize; }
    uint32_t get_logSumExpSize() { return logSumExpSize; }
    
    void set_s2(uint32_t s2) { this->s2 = s2; }
    void set_sInnerLoopSize(uint32_t sInnerLoopSize) { this->sInnerLoopSize = sInnerLoopSize; }
    void set_accumOutSize(uint32_t accumOutSize) { this->accumOutSize = accumOutSize; }
    void set_logSumExpSize(uint32_t logSumExpSize) { this->logSumExpSize = logSumExpSize; }
};

class IncreFlashAttentionTNDSplitCoreParamsMla {
public:
    uint32_t tndFDCoreArrLen; 
    uint32_t reserve;
    uint32_t balanceFDCoreBArr[MAX_AIC_CORE_NUM]; 
    uint32_t balanceFDCoreS1Arr[MAX_AIC_CORE_NUM];
    uint32_t balanceFDCoreKVSplitArr[MAX_AIC_CORE_NUM];
    uint32_t balanceFDCoreStartKVSplitNum[MAX_AIC_CORE_NUM];

    uint32_t get_tndFDCoreArrLen() { return tndFDCoreArrLen; }
    uint32_t get_reserve() { return reserve; }
    uint32_t* get_balanceFDCoreBArr() { return balanceFDCoreBArr; }
    uint32_t* get_balanceFDCoreS1Arr() { return balanceFDCoreS1Arr; }
    uint32_t* get_balanceFDCoreKVSplitArr() { return balanceFDCoreKVSplitArr; }
    uint32_t* get_balanceFDCoreStartKVSplitNum() { return balanceFDCoreStartKVSplitNum; }
    
    void set_tndFDCoreArrLen(uint32_t tndFDCoreArrLen) { this->tndFDCoreArrLen = tndFDCoreArrLen; }
    void set_reserve(uint32_t reserve) { this->reserve = reserve; }
};

class IncreFlashAttentionTilingDataMla {
public:
    IncreFlashAttentionBaseParamsMla baseParams;
    IncreFlashAttentionSplitKVParamsMla splitKVParams;
    IncreFlashAttentionTNDSplitCoreParamsMla tndSplitCoreParams;
    IncreFlashAttentionCoreParamsMla increFlashAttentionCoreParams;
    IncreFlashAttentionSingleCoreParamsMla increFlashAttentionSingleCoreParams;
    IncreFlashAttentionSingleCoreTensorSizeMla increFlashAttentionSingleCoreTensorSize;
};
// #endif

// } // namespace optiling
// #endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_H_