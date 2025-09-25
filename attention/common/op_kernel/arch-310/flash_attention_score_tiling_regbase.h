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
 * \file flash_attention_score_tiling_regbase.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_TILING_REGBASE_H_
#define FLASH_ATTENTION_SCORE_GRAD_TILING_REGBASE_H_

#include <cstdint>

namespace optiling {
class FlashAttentionScoreEmptyInputTilingDataRegbase {
public:
    uint32_t coreNum;
    uint32_t attentionOutFormerNum;
    uint32_t attentionOutTailNum;
    uint32_t softmaxMaxFormerNum;
    uint32_t softmaxMaxTailNum;
    uint64_t attentionOutSingleCoreDataSize;
    uint64_t attentionOutTailCoreDataSize;
    uint64_t softmaxMaxSingleCoreDataSize;
    uint64_t softmaxMaxTailCoreDataSize;
    uint64_t attentionOutLastCoreDataSize;
    uint64_t attentionOutLastCoreIndex;

    uint32_t get_coreNum() const {return coreNum;}
    uint32_t get_attentionOutFormerNum() const {return attentionOutFormerNum;}
    uint32_t get_attentionOutTailNum() const {return attentionOutTailNum;}
    uint32_t get_softmaxMaxFormerNum() const {return softmaxMaxFormerNum;}
    uint32_t get_softmaxMaxTailNum() const {return softmaxMaxTailNum;}
    uint64_t get_attentionOutSingleCoreDataSize() const {return attentionOutSingleCoreDataSize;}
    uint64_t get_attentionOutTailCoreDataSize() const {return attentionOutTailCoreDataSize;}
    uint64_t get_softmaxMaxSingleCoreDataSize() const {return softmaxMaxSingleCoreDataSize;}
    uint64_t get_softmaxMaxTailCoreDataSize() const {return softmaxMaxTailCoreDataSize;}
    uint64_t get_attentionOutLastCoreDataSize() const {return attentionOutLastCoreDataSize;}
    uint64_t get_attentionOutLastCoreIndex() const {return attentionOutLastCoreIndex;}

    void set_coreNum(uint32_t coreNum) {this->coreNum = coreNum;}
    void set_attentionOutFormerNum(uint32_t attentionOutFormerNum) {this->attentionOutFormerNum = attentionOutFormerNum;}
    void set_attentionOutTailNum(uint32_t attentionOutTailNum) {this->attentionOutTailNum = attentionOutTailNum;}
    void set_softmaxMaxFormerNum(uint32_t softmaxMaxFormerNum) {this->softmaxMaxFormerNum = softmaxMaxFormerNum;}
    void set_softmaxMaxTailNum(uint32_t softmaxMaxTailNum) {this->softmaxMaxTailNum = softmaxMaxTailNum;}
    void set_attentionOutSingleCoreDataSize(uint64_t attentionOutSingleCoreDataSize) {this->attentionOutSingleCoreDataSize = attentionOutSingleCoreDataSize;}
    void set_attentionOutTailCoreDataSize(uint64_t attentionOutTailCoreDataSize) {this->attentionOutTailCoreDataSize = attentionOutTailCoreDataSize;}
    void set_softmaxMaxSingleCoreDataSize(uint64_t softmaxMaxSingleCoreDataSize) {this->softmaxMaxSingleCoreDataSize = softmaxMaxSingleCoreDataSize;}
    void set_softmaxMaxTailCoreDataSize(uint64_t softmaxMaxTailCoreDataSize) {this->softmaxMaxTailCoreDataSize = softmaxMaxTailCoreDataSize;}
    void set_attentionOutLastCoreDataSize(uint64_t attentionOutLastCoreDataSize) {this->attentionOutLastCoreDataSize = attentionOutLastCoreDataSize;}
    void set_attentionOutLastCoreIndex(uint64_t attentionOutLastCoreIndex) {this->attentionOutLastCoreIndex = attentionOutLastCoreIndex;}
};

class InputParamsRegbase {
public:
    int64_t bSize;
    int64_t n2Size;
    int64_t gSize;
    int64_t s1Size;
    int64_t s2Size;
    int64_t alignedS2;
    int64_t dSize;
    int64_t dSizeV;
    int64_t dSizeRope;
    float keepProb;
    float scaleValue;
    int64_t preTokens;
    int64_t nextTokens;
    int64_t pseS1Size;
    int64_t pseS2Size;
    uint32_t pseBSize;
    uint32_t bandIndex;
    uint8_t layoutType;  // 1: BSH/BSND, 2: SBH, 3: BNSD
    uint8_t pseShapeType;  // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    uint8_t attenMaskShapeType;  // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
    uint8_t attenMaskDataType;  // 0: fp16, 1: bool(uint8)
    uint8_t attenMaskCompressMode;  // ALL: 0, NONE: 1, ANY: 2, CAUSAL: 3, BAND: 4 };
    uint8_t implMode;  // 0: high precise, 1: high performance, 2: invalid line high precise
    uint8_t sparseType;
    uint8_t needDropMaskOp;
    uint8_t dropMaskOuter;
    uint8_t pseEncodeType;
    uint16_t remain;
    uint32_t attenMaskS2Size;
    uint32_t pseType;
    uint32_t rsv1;
    int64_t qStartIdx;
    int64_t kvStartIdx;
    int64_t s1SparseValidSize;
    int64_t s2SparseValidSize;
    int64_t seed;
    int64_t offset;
    int64_t keepProbUint8;
    int64_t pseAlibiBaseS1;
    int64_t pseAlibiBaseS2;

    // PFA
    uint8_t deqScaleFlag;  // 0: uint64  1: float32
    uint8_t deqScale2Flag;  // 0: uint64  1: float32
    uint8_t isActualSeqLengthsNull;
    uint8_t isActualSeqLengthsKVNull;
    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;
    uint8_t isKvContinuous;
    uint8_t fromFused;
    uint8_t isBSNDOut;
    uint8_t isGqa;
    uint8_t isSoftMaxLseEnable;
    uint8_t isActualSharedPrefixLenNull;
    uint8_t isQHasLeftPadding;
    uint8_t isKVHasLeftPadding;
    uint32_t ropeHeadSize;
    uint32_t prefixSeqInnerSize;
    uint32_t headNumRatio;
    int32_t blockSize;
    int32_t blockTableDim2;
    int32_t paBlockNumSum;
    int32_t attenMaskS1Size;
    uint32_t kvSplitPart;
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
    uint8_t paLayoutType;
    uint8_t isRowInvalid;
    uint8_t isPostQuantPerChnl;
    uint8_t isPostQuantBF16;

    int64_t get_bSize() const {return bSize;}

    void set_bSize(int64_t bSize) {this->bSize = bSize;}

    int64_t get_n2Size() const {return n2Size;}

    void set_n2Size(int64_t n2Size) {this->n2Size = n2Size;}

    int64_t get_gSize() const {return gSize;}

    void set_gSize(int64_t gSize) {this->gSize = gSize;}

    int64_t get_s1Size() const {return s1Size;}

    void set_s1Size(int64_t s1Size) {this->s1Size = s1Size;}

    int64_t get_s2Size() const {return s2Size;}

    void set_s2Size(int64_t s2Size) {this->s2Size = s2Size;}

    int64_t get_alignedS2() const {return alignedS2;}

    void set_alignedS2(int64_t alignedS2) {this->alignedS2 = alignedS2;}

    int64_t get_dSize() const {return dSize;}

    void set_dSize(int64_t dSize) {this->dSize = dSize;}

    int64_t get_dSizeV() const {return dSizeV;}

    void set_dSizeV(int64_t dSizeV) {this->dSizeV = dSizeV;}

    int64_t get_dSizeRope() const {return dSizeRope;}

    void set_dSizeRope(int64_t dSizeRope) {this->dSizeRope = dSizeRope;}

    float get_keepProb() const {return keepProb;}

    void set_keepProb(float keepProb) {this->keepProb = keepProb;}

    float get_scaleValue() const {return scaleValue;}

    void set_scaleValue(float scaleValue) {this->scaleValue = scaleValue;}

    int64_t get_preTokens() const {return preTokens;}

    void set_preTokens(int64_t preTokens) {this->preTokens = preTokens;}

    int64_t get_nextTokens() const {return nextTokens;}

    void set_nextTokens(int64_t nextTokens) {this->nextTokens = nextTokens;}

    int64_t get_pseS1Size() const {return pseS1Size;}

    void set_pseS1Size(int64_t pseS1Size) {this->pseS1Size = pseS1Size;}

    int64_t get_pseS2Size() const {return pseS2Size;}

    void set_pseS2Size(int64_t pseS2Size) {this->pseS2Size = pseS2Size;}

    uint32_t get_pseBSize() const {return pseBSize;}

    void set_pseBSize(uint32_t pseBSize) {this->pseBSize = pseBSize;}

    uint32_t get_bandIndex() const {return bandIndex;}

    void set_bandIndex(uint32_t bandIndex) {this->bandIndex = bandIndex;}

    uint8_t get_layoutType() const {return layoutType;}

    void set_layoutType(uint8_t layoutType) {this->layoutType = layoutType;}

    uint8_t get_pseShapeType() const {return pseShapeType;}

    void set_pseShapeType(uint8_t pseShapeType) {this->pseShapeType = pseShapeType;}

    uint8_t get_attenMaskShapeType() const {return attenMaskShapeType;}

    void set_attenMaskShapeType(uint8_t attenMaskShapeType) {this->attenMaskShapeType = attenMaskShapeType;}

    uint8_t get_attenMaskDataType() const {return attenMaskDataType;}

    void set_attenMaskDataType(uint8_t attenMaskDataType) {this->attenMaskDataType = attenMaskDataType;}

    uint8_t get_attenMaskCompressMode() const {return attenMaskCompressMode;}

    void set_attenMaskCompressMode(uint8_t attenMaskCompressMode) {this->attenMaskCompressMode = attenMaskCompressMode;}

    uint8_t get_implMode() const {return implMode;}

    void set_implMode(uint8_t implMode) {this->implMode = implMode;}

    uint8_t get_sparseType() const {return sparseType;}

    void set_sparseType(uint8_t sparseType) {this->sparseType = sparseType;}

    uint8_t get_needDropMaskOp() const {return needDropMaskOp;}

    void set_needDropMaskOp(uint8_t needDropMaskOp) {this->needDropMaskOp = needDropMaskOp;}

    uint8_t get_dropMaskOuter() const {return dropMaskOuter;}

    void set_dropMaskOuter(uint8_t dropMaskOuter) {this->dropMaskOuter = dropMaskOuter;}

    uint8_t get_pseEncodeType() const {return pseEncodeType;}

    void set_pseEncodeType(uint8_t pseEncodeType) {this->pseEncodeType = pseEncodeType;}

    uint16_t get_remain() const {return remain;}

    void set_remain(uint16_t remain) {this->remain = remain;}

    uint32_t get_attenMaskS2Size() const {return attenMaskS2Size;}

    void set_attenMaskS2Size(uint32_t attenMaskS2Size) {this->attenMaskS2Size = attenMaskS2Size;}

    uint32_t get_pseType() const {return pseType;}

    void set_pseType(uint32_t pseType) {this->pseType = pseType;}

    uint32_t get_rsv1() const {return rsv1;}

    void set_rsv1(uint32_t rsv1) {this->rsv1 = rsv1;}

    int64_t get_qStartIdx() const {return qStartIdx;}

    void set_qStartIdx(int64_t qStartIdx) {this->qStartIdx = qStartIdx;}

    int64_t get_kvStartIdx() const {return kvStartIdx;}

    void set_kvStartIdx(int64_t kvStartIdx) {this->kvStartIdx = kvStartIdx;}

    int64_t get_s1SparseValidSize() const {return s1SparseValidSize;}

    void set_s1SparseValidSize(int64_t s1SparseValidSize) {this->s1SparseValidSize = s1SparseValidSize;}

    int64_t get_s2SparseValidSize() const {return s2SparseValidSize;}

    void set_s2SparseValidSize(int64_t s2SparseValidSize) {this->s2SparseValidSize = s2SparseValidSize;}

    int64_t get_seed() const {return seed;}

    void set_seed(int64_t seed) {this->seed = seed;}

    int64_t get_offset() const {return offset;}

    void set_offset(int64_t offset) {this->offset = offset;}

    int64_t get_keepProbUint8() const {return keepProbUint8;}

    void set_keepProbUint8(int64_t keepProbUint8) {this->keepProbUint8 = keepProbUint8;}

    int64_t get_pseAlibiBaseS1() const {return pseAlibiBaseS1;}

    void set_pseAlibiBaseS1(int64_t pseAlibiBaseS1) {this->pseAlibiBaseS1 = pseAlibiBaseS1;}

    int64_t get_pseAlibiBaseS2() const {return pseAlibiBaseS2;}

    void set_pseAlibiBaseS2(int64_t pseAlibiBaseS2) {this->pseAlibiBaseS2 = pseAlibiBaseS2;}

    uint8_t get_deqScaleFlag() const {return deqScaleFlag;}

    void set_deqScaleFlag(uint8_t deqScaleFlag) {this->deqScaleFlag = deqScaleFlag;}

    uint8_t get_deqScale2Flag() const {return deqScale2Flag;}

    void set_deqScale2Flag(uint8_t deqScale2Flag) {this->deqScale2Flag = deqScale2Flag;}

    uint8_t get_isActualSeqLengthsNull() const {return isActualSeqLengthsNull;}

    void set_isActualSeqLengthsNull(uint8_t isActualSeqLengthsNull) {this->isActualSeqLengthsNull = isActualSeqLengthsNull;}

    uint8_t get_isActualSeqLengthsKVNull() const {return isActualSeqLengthsKVNull;}

    void set_isActualSeqLengthsKVNull(uint8_t isActualSeqLengthsKVNull) {this->isActualSeqLengthsKVNull = isActualSeqLengthsKVNull;}

    uint32_t get_actualSeqLengthsSize() const {return actualSeqLengthsSize;}

    void set_actualSeqLengthsSize(uint32_t actualSeqLengthsSize) {this->actualSeqLengthsSize = actualSeqLengthsSize;}

    uint32_t get_actualSeqLengthsKVSize() const {return actualSeqLengthsKVSize;}

    void set_actualSeqLengthsKVSize(uint32_t actualSeqLengthsKVSize) {this->actualSeqLengthsKVSize = actualSeqLengthsKVSize;}

    uint8_t get_isKvContinuous() const {return isKvContinuous;}

    void set_isKvContinuous(uint8_t isKvContinuous) {this->isKvContinuous = isKvContinuous;}

    uint8_t get_fromFused() const {return fromFused;}

    void set_fromFused(uint8_t fromFused) {this->fromFused = fromFused;}

    uint8_t get_isBSNDOut() const {return isBSNDOut;}

    void set_isBSNDOut(uint8_t isBSNDOut) {this->isBSNDOut = isBSNDOut;}

    uint8_t get_isGqa() const {return isGqa;}

    void set_isGqa(uint8_t isGqa) {this->isGqa = isGqa;}

    uint8_t get_isSoftMaxLseEnable() const {return isSoftMaxLseEnable;}

    void set_isSoftMaxLseEnable(uint8_t isSoftMaxLseEnable) {this->isSoftMaxLseEnable = isSoftMaxLseEnable;}

    uint8_t get_isActualSharedPrefixLenNull() const {return isActualSharedPrefixLenNull;}

    void set_isActualSharedPrefixLenNull(uint8_t isActualSharedPrefixLenNull) {this->isActualSharedPrefixLenNull = isActualSharedPrefixLenNull;}

    uint8_t get_isQHasLeftPadding() const {return isQHasLeftPadding;}

    void set_isQHasLeftPadding(uint8_t isQHasLeftPadding) {this->isQHasLeftPadding = isQHasLeftPadding;}

    uint8_t get_isKVHasLeftPadding() const {return isKVHasLeftPadding;}

    void set_isKVHasLeftPadding(uint8_t isKVHasLeftPadding) {this->isKVHasLeftPadding = isKVHasLeftPadding;}

    uint32_t get_ropeHeadSize() const {return ropeHeadSize;}

    void set_ropeHeadSize(uint32_t ropeHeadSize) {this->ropeHeadSize = ropeHeadSize;}

    uint32_t get_prefixSeqInnerSize() const {return prefixSeqInnerSize;}

    void set_prefixSeqInnerSize(uint32_t prefixSeqInnerSize) {this->prefixSeqInnerSize = prefixSeqInnerSize;}

    uint32_t get_headNumRatio() const {return headNumRatio;}

    void set_headNumRatio(uint32_t headNumRatio) {this->headNumRatio = headNumRatio;}

    int32_t get_blockSize() const {return blockSize;}

    void set_blockSize(int32_t blockSize) {this->blockSize = blockSize;}

    int32_t get_blockTableDim2() const {return blockTableDim2;}

    void set_blockTableDim2(int32_t blockTableDim2) {this->blockTableDim2 = blockTableDim2;}

    int32_t get_paBlockNumSum() const {return paBlockNumSum;}

    void set_paBlockNumSum(int32_t paBlockNumSum) {this->paBlockNumSum = paBlockNumSum;}

    uint8_t get_paLayoutType() const {return paLayoutType;}

    void set_paLayoutType(uint8_t paLayoutType) {this->paLayoutType = paLayoutType;}

    uint32_t get_attenMaskS1Size() const {return attenMaskS1Size;}

    void set_attenMaskS1Size(uint32_t attenMaskS1Size) {this->attenMaskS1Size = attenMaskS1Size;}

    uint8_t get_isRowInvalid() const {return isRowInvalid;}

    void set_isRowInvalid(uint8_t isRowInvalid) {this->isRowInvalid = isRowInvalid;}

    uint8_t get_isPostQuantPerChnl() const {return isPostQuantPerChnl;}

    void set_isPostQuantPerChnl(uint8_t isPostQuantPerChnl) {this->isPostQuantPerChnl = isPostQuantPerChnl;}

    uint8_t get_isPostQuantBF16() const {return isPostQuantBF16;}

    void set_isPostQuantBF16(uint8_t isPostQuantBF16) {this->isPostQuantBF16 = isPostQuantBF16;}
};

class MultiCoreParamsRegbase {
public:
    int32_t coreNum;
    int64_t totalSize;
    int64_t s1OuterSize;
    int64_t splitFactorSize;
    int64_t splitFactorTailSize;
    uint32_t bnStartIdx[48];
    int64_t sparseStartIdx[48];

    int32_t get_coreNum() const {return coreNum;}

    void set_coreNum(int32_t coreNum) {this->coreNum = coreNum;}

    int64_t get_totalSize() const {return totalSize;}

    void set_totalSize(int64_t totalSize) {this->totalSize = totalSize;}

    int64_t get_s1OuterSize() const {return s1OuterSize;}

    void set_s1OuterSize(int64_t s1OuterSize) {this->s1OuterSize = s1OuterSize;}

    int64_t get_splitFactorSize() const {return splitFactorSize;}

    void set_splitFactorSize(int64_t splitFactorSize) {this->splitFactorSize = splitFactorSize;}

    int64_t get_splitFactorTailSize() const {return splitFactorTailSize;}

    void set_splitFactorTailSize(int64_t splitFactorTailSize) {this->splitFactorTailSize = splitFactorTailSize;}

    uint32_t *get_bnStartIdxPtr() {return bnStartIdx;}

    int64_t *get_sparseStartIdxPtr() {return sparseStartIdx;}
};

class DropmaskParamsRegbase {
public:
    int32_t multiCoreFactorSize;
    int32_t baseUbCalSize;
    int64_t multiCoreTotalSize;
    int64_t shapeTotalSize;

    int32_t get_multiCoreFactorSize() const {return multiCoreFactorSize;}

    void set_multiCoreFactorSize(int32_t multiCoreFactorSize) {this->multiCoreFactorSize = multiCoreFactorSize;}

    int32_t get_baseUbCalSize() const {return baseUbCalSize;}

    void set_baseUbCalSize(int32_t baseUbCalSize) {this->baseUbCalSize = baseUbCalSize;}

    int64_t get_multiCoreTotalSize() const {return multiCoreTotalSize;}

    void set_multiCoreTotalSize(int64_t multiCoreTotalSize) {this->multiCoreTotalSize = multiCoreTotalSize;}

    int64_t get_shapeTotalSize() const {return shapeTotalSize;}

    void set_shapeTotalSize(int64_t shapeTotalSize) {this->shapeTotalSize = shapeTotalSize;}
};

class InitOutputParams {
public:
    uint32_t singleCoreSize;
    uint8_t needInit;
    uint8_t isOneN;
    uint8_t rsvd[2];
    int64_t totalOutputSize;
    int64_t totalSoftMaxLseOutputSize;

    uint32_t get_singleCoreSize() const {return singleCoreSize;}

    void set_singleCoreSize(uint32_t singleCoreSize) {this->singleCoreSize = singleCoreSize;}

    uint8_t get_needInit() const {return needInit;}

    void set_needInit(uint8_t needInit) {this->needInit = needInit;}

    uint8_t get_isOneN() const {return isOneN;}

    void set_isOneN(uint8_t isOneN) {this->isOneN = isOneN;}

    uint8_t *get_rsvdPtr() {return rsvd;}

    int64_t get_totalOutputSize() const {return totalOutputSize;}

    void set_totalOutputSize(int64_t totalOutputSize) {this->totalOutputSize = totalOutputSize;}

    int64_t get_totalSoftMaxLseOutputSize() const {return totalSoftMaxLseOutputSize;}

    void set_totalSoftMaxLseOutputSize(int64_t totalSoftMaxLseOutputSize) {this->totalSoftMaxLseOutputSize = totalSoftMaxLseOutputSize;}
};

class FlashAttentionScoreSimplifiedTilingData {
public:
    InputParamsRegbase inputParamsRegbase;
    MultiCoreParamsRegbase multiCoreParamsRegbase;
    DropmaskParamsRegbase dropmaskParamsRegbase;
    InitOutputParams initOutputParams;
};
}  // namespace optiling
#endif