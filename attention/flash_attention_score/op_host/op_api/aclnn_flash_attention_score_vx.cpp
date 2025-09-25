/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_flash_attention_score_vx.h"
#include "flash_attention_score.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;
static const int64_t HEAD_DIM_MAX = 512;
static const int64_t PSE_TYPE_V1 = 1; // add and mul
static const int64_t PSE_INNER_MUL_ADD = 2;
static const int64_t PSE_INNER_MUL_ADD_SQRT = 3;

struct AxesInfo {
    int64_t b;
    int64_t n1;
    int64_t n2;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dk;
    int64_t dv;
};

enum class InputLayout : int64_t {
    BSND,
    SBH,
    BNSD,
    BSH,
    TND
};

struct FaShapeInfo {
    AxesInfo axes;
    InputLayout inputLayout;
    string l0InputLayoutStr;
    uint64_t dimNum = 0;
};

struct FaTensorInput {
    const aclTensor *query = nullptr;
    const aclTensor *key = nullptr;
    const aclTensor *value = nullptr;
    const aclTensor *realShiftOptional = nullptr;
    const aclTensor *dropMaskOptional = nullptr;
    const aclTensor *paddingMaskOptional = nullptr;
    const aclTensor *attenMaskOptional = nullptr;
    const aclTensor *dScaleQOptional = nullptr;
    const aclTensor *dScaleKOptional = nullptr;
    const aclTensor *dScaleVOptional = nullptr;
    const aclTensor *queryRopeOptional = nullptr;
    const aclTensor *keyRopeOptional = nullptr;
};

struct FaTensorOutput {
    const aclTensor *softmaxMaxOut= nullptr;
    const aclTensor *softmaxSumOut= nullptr;
    const aclTensor *softmaxOutOut= nullptr;
    const aclTensor *attentionOutOut= nullptr;
};

static void AnalysisAxisForBsh(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BSH;
    shapeInfo.l0InputLayoutStr = "BSH";
    uint64_t dSize = qShape[2] / shapeInfo.axes.n1;
    shapeInfo.axes.d = dSize;
    if (dSize == 0) {
        return;
    }
    shapeInfo.axes.b = qShape[0];
    shapeInfo.axes.n2 = kShape[DIM_NUM_2] / dSize;
    shapeInfo.axes.s1 = qShape[1];
    shapeInfo.axes.s2 = kShape[1];
    shapeInfo.axes.dk = kShape[DIM_NUM_2] / shapeInfo.axes.n2;
    shapeInfo.axes.dv = vShape[DIM_NUM_2] / shapeInfo.axes.n2;
}

static void AnalysisAxisForBsnd(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BSND;
    shapeInfo.l0InputLayoutStr = "BSND";
    shapeInfo.axes.b = qShape[0];
    shapeInfo.axes.n2 = kShape[DIM_NUM_2];
    shapeInfo.axes.s1 = qShape[1];
    shapeInfo.axes.s2 = kShape[1];
    shapeInfo.axes.d = qShape[DIM_NUM_3];
    shapeInfo.axes.dk = kShape[DIM_NUM_3];
    shapeInfo.axes.dv = vShape[DIM_NUM_3];
}

static void AnalysisAxisForTnd(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::TND;
    shapeInfo.l0InputLayoutStr = "TND";
    shapeInfo.axes.n2 = kShape[1];
    shapeInfo.axes.d = qShape[DIM_NUM_2];
    shapeInfo.axes.dk = kShape[DIM_NUM_2];
    shapeInfo.axes.dv = vShape[DIM_NUM_2];
}

static void AnalysisAxisForSbh(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::SBH;
    shapeInfo.l0InputLayoutStr = "SBH";
    uint64_t dSize = qShape[2] / shapeInfo.axes.n1;
    shapeInfo.axes.d = dSize;
    if (dSize == 0) {
        return;
    }
    shapeInfo.axes.b = qShape[1];
    shapeInfo.axes.n2 = kShape[DIM_NUM_2] / dSize;
    shapeInfo.axes.s1 = qShape[0];
    shapeInfo.axes.s2 = kShape[0];
    shapeInfo.axes.dk = kShape[DIM_NUM_2] / shapeInfo.axes.n2;
    shapeInfo.axes.dv = vShape[DIM_NUM_2] / shapeInfo.axes.n2;
}

static void AnalysisAxisForBnsd(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BNSD;
    shapeInfo.l0InputLayoutStr = "BNSD";
    shapeInfo.axes.b = qShape[0];
    shapeInfo.axes.n2 = kShape[1];
    shapeInfo.axes.s1 = qShape[DIM_NUM_2];
    shapeInfo.axes.s2 = kShape[DIM_NUM_2];
    shapeInfo.axes.d = qShape[DIM_NUM_3];
    shapeInfo.axes.dk = kShape[DIM_NUM_3];
    shapeInfo.axes.dv = vShape[DIM_NUM_3];
}

static aclnnStatus AnalysisAxis(FaTensorInput &faTensorInput, const char *inputLayout, int64_t headNum,
                                FaShapeInfo &shapeInfo)
{
    Shape qShape = faTensorInput.query->GetViewShape();
    Shape kShape = faTensorInput.key->GetViewShape();
    Shape vShape = faTensorInput.value->GetViewShape();
    shapeInfo.dimNum = qShape.GetDimNum();

    // 记录轴的长度 b, n2, g, s1, s2, d
    // H1等于N1*D, H2等于N2*D
    // N1等于g*N2
    shapeInfo.axes.n1 = headNum;
    std::string inputLayoutStr = op::ToString(inputLayout).GetString();
    if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "BSH") {
        // query: (B,S1,N1*D)
        // key/value: (B,S2,N2*D)
        AnalysisAxisForBsh(qShape, kShape, vShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_4 && inputLayoutStr == "BSND") {
        // query: (B,S1,N1,D)
        // key/value: (B,S2,N2,D)
        AnalysisAxisForBsnd(qShape, kShape, vShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "SBH") {
        // query: (S1,B,N1*D)
        // key/value: (S2,B,N2*D)
        AnalysisAxisForSbh(qShape, kShape, vShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_4 && inputLayoutStr == "BNSD") {
        // query: (B,N1,S1,D)
        // key/value: (B,N2,S2,D)
        AnalysisAxisForBnsd(qShape, kShape, vShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "TND") {
        // query: (T,N1,D)
        // key/value: (T,N2,D)
        AnalysisAxisForTnd(qShape, kShape, vShape, shapeInfo);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not support input_layout %s with dim_num %lu", inputLayout, shapeInfo.dimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.d != shapeInfo.axes.dk) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qD and kD should be same, but got qD=%ld kD=%ld", shapeInfo.axes.d, 
            shapeInfo.axes.dk);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.d < shapeInfo.axes.dv) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support kD >= vD, but got kD=%ld vD=%ld", shapeInfo.axes.d, 
            shapeInfo.axes.dv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus InputDtypeCheck(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                   const aclTensor *attentionOut, const aclTensor *realShiftOptional, int64_t pseType)
{
    auto vDtype = value->GetDataType();
    auto kDtype = key->GetDataType();
    auto qDtype = query->GetDataType();
    auto outDtype = attentionOut->GetDataType();
    if (qDtype != kDtype || kDtype != vDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of query[%s], key[%s], value[%s] are not equal.",
                op::ToString(DataType(qDtype)).GetString(), op::ToString(DataType(kDtype)).GetString(),
                op::ToString(DataType(vDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (pseType == PSE_INNER_MUL_ADD || pseType == PSE_INNER_MUL_ADD_SQRT) {
        // Inner pse alibi, dtype must be fp32
        if (realShiftOptional == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When pseType is 2 or 3, pseShape cannot be null.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        auto pseDtype = realShiftOptional->GetDataType();
        if (pseDtype != op::DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The data type %s of pse is not invalid in pse type 2 or 3 mode, It must be float32",
                    op::ToString(DataType(pseDtype)).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
    if (realShiftOptional != nullptr) {
        auto pseDtype = realShiftOptional->GetDataType();
        if (pseDtype != outDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The data type %s of pse is not equal to the data type %s of attentionOut.",
                    op::ToString(DataType(pseDtype)).GetString(), op::ToString(DataType(outDtype)).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus AnalysisInput(FaTensorInput &faTensorInput, char *inputLayout, int64_t headNum,
                                 FaShapeInfo &shapeInfo)
{
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "head_num must > 0, but got %ld", headNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(
        AnalysisAxis(faTensorInput, inputLayout, headNum, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (shapeInfo.axes.d > HEAD_DIM_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Head dim must <= 512, but got %ld", shapeInfo.axes.d);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus Contiguous(FaTensorInput &faTensorInput, aclOpExecutor *executor)
{
    faTensorInput.query = l0op::Contiguous(faTensorInput.query, executor);
    CHECK_RET(faTensorInput.query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    faTensorInput.key = l0op::Contiguous(faTensorInput.key, executor);
    CHECK_RET(faTensorInput.key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    faTensorInput.value = l0op::Contiguous(faTensorInput.value, executor);
    CHECK_RET(faTensorInput.value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (faTensorInput.realShiftOptional) {
        faTensorInput.realShiftOptional = l0op::Contiguous(faTensorInput.realShiftOptional, executor);
        CHECK_RET(faTensorInput.realShiftOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (faTensorInput.dropMaskOptional) {
        faTensorInput.dropMaskOptional = l0op::Contiguous(faTensorInput.dropMaskOptional, executor);
        CHECK_RET(faTensorInput.dropMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (faTensorInput.paddingMaskOptional) {
        faTensorInput.paddingMaskOptional = l0op::Contiguous(faTensorInput.paddingMaskOptional, executor);
        CHECK_RET(faTensorInput.paddingMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (faTensorInput.attenMaskOptional) {
        faTensorInput.attenMaskOptional = l0op::Contiguous(faTensorInput.attenMaskOptional, executor);
        CHECK_RET(faTensorInput.attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (faTensorInput.queryRopeOptional) {
        faTensorInput.queryRopeOptional = l0op::Contiguous(faTensorInput.queryRopeOptional, executor);
        CHECK_RET(faTensorInput.queryRopeOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (faTensorInput.keyRopeOptional) {
        faTensorInput.keyRopeOptional = l0op::Contiguous(faTensorInput.keyRopeOptional, executor);
        CHECK_RET(faTensorInput.keyRopeOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (faTensorInput.dScaleQOptional) {
        faTensorInput.dScaleQOptional = l0op::Contiguous(faTensorInput.dScaleQOptional, executor);
        CHECK_RET(faTensorInput.dScaleQOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (faTensorInput.dScaleKOptional) {
        faTensorInput.dScaleKOptional = l0op::Contiguous(faTensorInput.dScaleKOptional, executor);
        CHECK_RET(faTensorInput.dScaleKOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (faTensorInput.dScaleVOptional) {
        faTensorInput.dScaleVOptional = l0op::Contiguous(faTensorInput.dScaleVOptional, executor);
        CHECK_RET(faTensorInput.dScaleVOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFaParam(
    const FaTensorInput &faTensorInput, const char *inputLayout,
    const FaTensorOutput &faTensorOutput, const uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 必须的参数指针判空
    CHECK_RET(faTensorInput.query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(faTensorInput.key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(faTensorInput.value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(inputLayout != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(faTensorOutput.softmaxMaxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(faTensorOutput.softmaxSumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(faTensorOutput.attentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreVXGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional, const aclTensor *dScaleQOptional,
    const aclTensor *dScaleKOptional, const aclTensor *dScaleVOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout,
    int64_t innerPrecise, int64_t sparseMode, int64_t outDtype, int64_t pseType, int64_t seed, int64_t offset,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    FaTensorInput faTensorInput = {query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                                   attenMaskOptional, dScaleQOptional, dScaleKOptional, dScaleVOptional,
                                   queryRopeOptional, keyRopeOptional};
    FaTensorOutput faTensorOutput = {softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut};
    CHECK_RET(CheckFaParam(faTensorInput, inputLayout,faTensorOutput, workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(aclnnFlashAttentionScoreVX,
                   DFX_IN(faTensorInput.query, faTensorInput.key, faTensorInput.value, faTensorInput.realShiftOptional,
                          faTensorInput.dropMaskOptional, faTensorInput.paddingMaskOptional,
                          faTensorInput.attenMaskOptional, faTensorInput.queryRopeOptional,
                          faTensorInput.keyRopeOptional, faTensorInput.dScaleQOptional,
                          faTensorInput.dScaleKOptional, faTensorInput.dScaleVOptional,
                          prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
                          qStartIdxOptional, kvStartIdxOptional, scaleValue, keepProb, preTokens, nextTokens, headNum,
                          inputLayout, innerPrecise, sparseMode, outDtype, pseType,
                          seed, offset),
                   DFX_OUT(faTensorOutput.softmaxMaxOut, faTensorOutput.softmaxSumOut, faTensorOutput.softmaxOutOut,
                           faTensorOutput.attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (faTensorOutput.softmaxMaxOut->IsEmpty() && faTensorOutput.softmaxSumOut->IsEmpty() &&
        faTensorOutput.attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    FaShapeInfo shapeInfo;
    CHECK_RET(InputDtypeCheck(faTensorInput.query, faTensorInput.key, faTensorInput.value,
                  faTensorOutput.attentionOutOut, faTensorInput.realShiftOptional, pseType) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(AnalysisInput(faTensorInput, inputLayout, headNum, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(Contiguous(faTensorInput, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto l0FlashAttentionScoreOuts = l0op::FlashAttentionScore(
        faTensorInput.query, faTensorInput.key, faTensorInput.value, faTensorInput.realShiftOptional,
        faTensorInput.dropMaskOptional, faTensorInput.paddingMaskOptional, faTensorInput.attenMaskOptional,
        prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional,
        faTensorInput.dScaleQOptional, faTensorInput.dScaleKOptional, faTensorInput.dScaleVOptional,
        faTensorInput.queryRopeOptional, faTensorInput.keyRopeOptional,
        scaleValue, keepProb, preTokens, nextTokens, headNum, shapeInfo.l0InputLayoutStr.c_str(),
        innerPrecise, sparseMode, pseType, seed, offset, outDtype, "", l0Executor);

    auto l0SoftmaxMaxOut = l0FlashAttentionScoreOuts[0];
    auto l0SoftmaxSumOut = l0FlashAttentionScoreOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FlashAttentionScoreOuts[3];

    if (l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr || l0AttentionOutOut == nullptr) {
      OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "l0SoftmaxMaxOut or l0SoftmaxSumOut or l0AttentionOutOut is null");
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_ERR_INNER_NULLPTR;
    }
    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, faTensorOutput.softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, faTensorOutput.softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, faTensorOutput.attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreVX(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreVX);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}  // namespace

#ifdef __cplusplus
}
#endif
