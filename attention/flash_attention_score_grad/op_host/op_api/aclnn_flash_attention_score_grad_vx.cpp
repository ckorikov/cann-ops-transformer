/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_flash_attention_score_grad_vx.h"
#include "flash_attention_score_grad.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/fast_vector.h"
#include "runtime/context.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
#define CHECK_SCALAR_TENSOR(condition)                                                                                             \
    do {                                                                                                                           \
        if (condition) {                                                                                                           \
            OP_LOGW("There is a scalar tensor in the input optional parameters, and we will treat this input parameter as null."); \
        }                                                                                                                          \
    } while (0)

typedef struct FagInShapeInfoS {
    int64_t n1Dim;
    int64_t n2Dim;
    int64_t h1Dim;
    int64_t h2Dim;
    int64_t s1Dim;
    int64_t s2Dim;
    int64_t dDim;
    int64_t dkDim;
    int64_t dvDim;
    int64_t alignDim;

    int64_t querySDimStrideSize;
    int64_t kvSDimStrideSize;

    std::string inputLayoutStr;

    bool needPadDimD;
    bool needTranspose;
    bool passThrowInnerFag;
    bool needBackwordReshape;
    bool needPadValueD;
} FagInShapeInfo;

typedef struct FagShapeArrayS {
    aclIntArray *queryShapeArray = nullptr;
    aclIntArray *keyShapeArray = nullptr;
    aclIntArray *dqShapeArray = nullptr;
    aclIntArray *dkShapeArray = nullptr;
    aclIntArray *queryBwShapeArray = nullptr;
    aclIntArray *keyBwShapeArray = nullptr;
    aclIntArray *dqBwShapeArray = nullptr;
    aclIntArray *dkBwShapeArray = nullptr;
    aclIntArray *valueReshapeBefore = nullptr;
    aclIntArray *valueReshapeAfter = nullptr;
    aclIntArray *attenInReshapeBefore = nullptr;
    aclIntArray *attenInReshapeAfter = nullptr;
    aclIntArray *dvReshapeBefore = nullptr;
    aclIntArray *dvReshapeAfter = nullptr;
} FagShapeArray;

typedef struct FagShapeCollectionS {
    const aclTensor *query = nullptr;
    const aclTensor *key = nullptr;
    const aclTensor *value = nullptr;
    const aclTensor *dy = nullptr;
    const aclTensor *attentionIn = nullptr;
    int64_t headNum = 0;
    const char *inputLayout = nullptr;
    FagInShapeInfo *fagShape = nullptr;
    const aclIntArray *actualSeqQLenOptional = nullptr;
    const aclIntArray *actualSeqKvLenOptional = nullptr;
    double keepProb = 0.0;
} FagShapeCollection;

typedef struct FagTensorInputS {
    const aclTensor *query = nullptr;
    const aclTensor *key = nullptr;
    const aclTensor *value = nullptr;
    const aclTensor *dy = nullptr;
    const aclTensor *attentionInOptional = nullptr;
    const aclTensor **queryCngs = nullptr;
    const aclTensor **keyCngs = nullptr;
    const aclTensor **valueCngs = nullptr;
    const aclTensor **dyCngs = nullptr;
    const aclTensor **attentionInOptionalCngs = nullptr;
} FagTensorInput;

static constexpr int64_t ALIGN_D_DIM_SIZE = 128;
static constexpr int64_t SPARE_ALIGN_D_DIM_SIZE = 16;
static constexpr int64_t MAX_BSN_DIMS_SIZE = 65535;
static constexpr int64_t MAX_LAYOUT_SIZE = 5;
static constexpr int64_t PSE_TYPE_V1 = 1; // add and mul
static const int64_t HEAD_DIM_8 = 8;
static const int64_t HEAD_DIM_72 = 72;
static const int64_t HEAD_DIM_88 = 88;
static const int64_t HEAD_DIM_192 = 192;
static const int64_t SEQ_LEN_4096 = 4096;
static constexpr size_t MIN_DIM = 3;
static const int64_t TND_MAX_S2 = 1024;
static const int64_t TND_MAX_S1_SUM = 160 * 1024;
static const int64_t TND_MAX_DDIM = 96;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DQ_OUT_IDX = 0;
static const uint64_t DK_OUT_IDX = 1;
static const uint64_t DV_OUT_IDX = 2;

char defaultSoftmaxInLayoutRegbase[] = "";

static aclnnStatus InvalidTensorDimCheck(FagShapeCollection &shapeCollect, const aclTensor *dq,
                                         const aclTensor *dk, const aclTensor *dv)
{
    auto queryDimNum = shapeCollect.query->GetViewShape().GetDimNum();
    auto keyDimNum = shapeCollect.key->GetViewShape().GetDimNum();
    auto valueDimNum = shapeCollect.value->GetViewShape().GetDimNum();
    auto dyDimNum = shapeCollect.dy->GetViewShape().GetDimNum();
    auto attentionInDimNum = shapeCollect.attentionIn->GetViewShape().GetDimNum();
    auto dqDimNum = dq->GetViewShape().GetDimNum();
    auto dkDimNum = dk->GetViewShape().GetDimNum();
    auto dvDimNum = dv->GetViewShape().GetDimNum();
    if (queryDimNum < MIN_DIM || keyDimNum < MIN_DIM || valueDimNum < MIN_DIM || dyDimNum < MIN_DIM ||
        attentionInDimNum < MIN_DIM || dqDimNum < MIN_DIM || dkDimNum < MIN_DIM || dvDimNum < MIN_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The input or output of FAG does not support tensors with dim less than 3.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus GetDiffDimInfo(const aclTensor *query, const aclTensor *key, const aclTensor *value,
    FagInShapeInfo &fagShape, int64_t headNum)
{
    auto queryShape = query->GetViewShape();
    auto keyShape = key->GetViewShape();
    auto valueShape = value->GetViewShape();
    if (fagShape.inputLayoutStr == "BSH" || fagShape.inputLayoutStr == "SBH") {
        fagShape.h1Dim = queryShape.GetDim(DIM_NUM_2); // 2:h1
        fagShape.h2Dim = keyShape.GetDim(DIM_NUM_2);    // 2:h2
        if (headNum == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The headNum is zero.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        fagShape.dDim = fagShape.h1Dim / headNum; // q Head-dim
        if (fagShape.dDim == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of D is zero.");
            return ACLNN_ERR_PARAM_INVALID;
        }

        fagShape.n1Dim = headNum;
        fagShape.n2Dim = fagShape.h2Dim / fagShape.dDim;
        fagShape.s1Dim = (fagShape.inputLayoutStr == "BSH") ? queryShape.GetDim(1) : queryShape.GetDim(0);
        fagShape.s2Dim = (fagShape.inputLayoutStr == "BSH") ? keyShape.GetDim(1) : keyShape.GetDim(0);
        fagShape.dkDim = keyShape.GetDim(DIM_NUM_2) / fagShape.n2Dim;
        fagShape.dvDim = valueShape.GetDim(DIM_NUM_2) / fagShape.n2Dim;
    } else if (fagShape.inputLayoutStr == "TND") {
        fagShape.dDim = queryShape.GetDim(DIM_NUM_2);  // 2:d
        fagShape.n1Dim = queryShape.GetDim(1); // 1:n1
        fagShape.n2Dim = keyShape.GetDim(1);    // 1:n2
        fagShape.dkDim = keyShape.GetDim(DIM_NUM_2);
        fagShape.dvDim = valueShape.GetDim(DIM_NUM_2);
    } else if (queryShape.GetDimNum() > MIN_DIM) {
        fagShape.dDim = queryShape.GetDim(DIM_NUM_3); // 3:d
        fagShape.dkDim = keyShape.GetDim(DIM_NUM_3); // key Head-dim
        fagShape.dvDim = valueShape.GetDim(DIM_NUM_3); // value Head-dim
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of the tensor whose input is BNSD/BSND is less than 4.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (fagShape.dDim != fagShape.dkDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qD and kD should be same, but got qD=%ld kD=%ld", fagShape.dDim, fagShape.dkDim);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (fagShape.dDim < fagShape.dvDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The headDim of value should be smaller than headDim of key.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus GetInputShapeInfo(FagShapeCollection &shapeCollect)
{
    auto queryShape = shapeCollect.query->GetViewShape();
    auto keyShape = shapeCollect.key->GetViewShape();
    auto queryDimSize = shapeCollect.query->Size();
    auto kvDimSize = shapeCollect.key->Size();
    int64_t headNum = shapeCollect.headNum;
    const char *inputLayout = shapeCollect.inputLayout;
    FagInShapeInfo &fagShape = *(shapeCollect.fagShape);
    fagShape.inputLayoutStr = op::ToString(inputLayout).GetString();
    fagShape.n1Dim = (fagShape.inputLayoutStr == "BNSD") ? queryShape.GetDim(1) : queryShape.GetDim(2); // 1 or 2:n1
    fagShape.n2Dim = (fagShape.inputLayoutStr == "BNSD") ? keyShape.GetDim(1) : keyShape.GetDim(2);       // 1 or 2:n2
    fagShape.s1Dim = (fagShape.inputLayoutStr == "BNSD") ? queryShape.GetDim(DIM_NUM_2) : queryShape.GetDim(1); // 1 or 2:s1
    fagShape.s2Dim = (fagShape.inputLayoutStr == "BNSD") ? keyShape.GetDim(DIM_NUM_2) : keyShape.GetDim(1);       // 1 or 2:s2

    auto ret = GetDiffDimInfo(shapeCollect.query, shapeCollect.key, shapeCollect.value, fagShape, headNum);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    fagShape.needPadValueD = (fagShape.dDim != fagShape.dvDim);

    fagShape.querySDimStrideSize = 0;
    fagShape.kvSDimStrideSize = 0;
    if (fagShape.inputLayoutStr == "BSND") { // stride is N * D
        fagShape.querySDimStrideSize = fagShape.n1Dim * fagShape.dDim;
        fagShape.kvSDimStrideSize = fagShape.n2Dim * fagShape.dDim;
    } else if (fagShape.inputLayoutStr == "BSH") {           // stride is H
        fagShape.querySDimStrideSize = queryShape.GetDim(DIM_NUM_2); // 2:dv
        fagShape.kvSDimStrideSize = keyShape.GetDim(DIM_NUM_2);       // 2:dv
    } else if (fagShape.inputLayoutStr == "SBH") {           // stride is B * H
        fagShape.querySDimStrideSize = fagShape.s1Dim == 0 ? 0 : (queryDimSize / fagShape.s1Dim);
        fagShape.kvSDimStrideSize = fagShape.s2Dim == 0 ? 0 : (kvDimSize / fagShape.s2Dim);
    }

    fagShape.alignDim = (fagShape.dDim < ALIGN_D_DIM_SIZE) ? SPARE_ALIGN_D_DIM_SIZE : ALIGN_D_DIM_SIZE;

    fagShape.needPadDimD = false;
    fagShape.needTranspose = false;

    fagShape.passThrowInnerFag = (!(fagShape.needPadDimD) && !(fagShape.needTranspose));
    fagShape.needBackwordReshape =
        (fagShape.inputLayoutStr == "SBH" && fagShape.needPadDimD && !(fagShape.needTranspose));
    return ACLNN_SUCCESS;
}

static inline aclnnStatus ContiguousTensorWithCheck(const aclTensor *inputTensor, const aclTensor **outTensor,
                                                    aclOpExecutor *executor)
{
    if (inputTensor != nullptr && inputTensor->GetViewShape().GetDimNum() != 0) {
        *outTensor = l0op::Contiguous(inputTensor, executor);
        CHECK_RET(*outTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 输入入参如果是标量tensor，将会按照此可选输入为null处理 ;
    CHECK_SCALAR_TENSOR(inputTensor != nullptr && inputTensor->GetViewShape().GetDimNum() == 0);

    return ACLNN_SUCCESS;
}

static inline void ConvertInputLayout(FagInShapeInfo fagShape, const char *inputLayout, char *inputLayoutUnderTrans,
                                      size_t layoutUnderTransSize)
{
    if (fagShape.needTranspose) {                  // 1. 只要是需要transpose，输入FAG layout必然是BNSD
        inputLayoutUnderTrans[0] = 'B';            // 0: 'B'
        inputLayoutUnderTrans[1] = 'N';            // 1: 'N'
        inputLayoutUnderTrans[2] = 'S';            // 2: 'S'
        inputLayoutUnderTrans[3] = 'D';            // 3: 'D'
    } else if (fagShape.needBackwordReshape) {     // 2. 如果是SBH仅PAD场景，输入FAG layout必然还是SBH
        inputLayoutUnderTrans[0] = inputLayout[0]; // 0: 'S'
        inputLayoutUnderTrans[1] = inputLayout[1]; // 1: 'B'
        inputLayoutUnderTrans[2] = 'H';            // 2: 'H'
    } else if (fagShape.needPadDimD) { // 3. 如果是仅PAD场景，根据BSH/SBH/BNSD/BSND自适应reshape后的layout
        /* BSH  -> BSND
           SBH  -> SBND
           TND  -> TND
           BNSD -> BNSD
           BSND -> BSND */
        for (size_t i = 0; i < strlen(inputLayout) && i < layoutUnderTransSize - 1; i++) {
            if (inputLayout[i] == 'H') {
                inputLayoutUnderTrans[i] = 'N';
                inputLayoutUnderTrans[i + 1] = 'D';
                break;
            }
            inputLayoutUnderTrans[i] = inputLayout[i];
        }
    } else { // 4. 其他情况，保持原始layout
        for (size_t i = 0; i < strlen(inputLayout) && i < layoutUnderTransSize - 1; i++) {
            inputLayoutUnderTrans[i] = inputLayout[i];
        }
    }
}

static aclnnStatus ContiguousInputTensor(FagTensorInput &tensorInput, aclOpExecutor *executor)
{
    auto ret = ACLNN_SUCCESS;

    // query如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(tensorInput.query, tensorInput.queryCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // key如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(tensorInput.key, tensorInput.keyCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // value如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(tensorInput.value, tensorInput.valueCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // dy如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(tensorInput.dy, tensorInput.dyCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // attentionInOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(tensorInput.attentionInOptional, tensorInput.attentionInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    return ret;
}

static aclnnStatus ContiguousOptionalInputTensor(
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
    const aclTensor *dScaleQOptional, const aclTensor *dScaleKOptional,
    const aclTensor *dScaleVOptional, const aclTensor *dScaleDyOptional, const aclTensor *dScaleOOptional,
    const aclTensor **pseShiftOptionalCngs, const aclTensor **dropMaskOptionalCngs,
    const aclTensor **paddingMaskOptionalCngs, const aclTensor **attenMaskOptionalCngs,
    const aclTensor **softmaxMaxOptionalCngs, const aclTensor **softmaxSumOptionalCngs,
    const aclTensor **softmaxInOptionalCngs, const aclTensor **queryRopeOptionalCngs,
    const aclTensor **keyRopeOptionalCngs, const aclTensor **dScaleQOptionalCngs,
    const aclTensor **dScaleKOptionalCngs, const aclTensor **dScaleVOptionalCngs, const aclTensor **dScaleDyOptionalCngs,
    const aclTensor **dScaleOOptionalCngs, aclOpExecutor *executor)
{
    auto ret = ACLNN_SUCCESS;

    // pseShiftOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(pseShiftOptional, pseShiftOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // dropMaskOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(dropMaskOptional, dropMaskOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // paddingMaskOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(paddingMaskOptional, paddingMaskOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // attenMaskOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(attenMaskOptional, attenMaskOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // softmaxMaxOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(softmaxMaxOptional, softmaxMaxOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // softmaxSumOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(softmaxSumOptional, softmaxSumOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // softmaxInOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(softmaxInOptional, softmaxInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // queryRopeOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(queryRopeOptional, queryRopeOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // keyRopeOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(keyRopeOptional, keyRopeOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // dScaleQOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(dScaleQOptional, dScaleQOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // dScaleKOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(dScaleKOptional, dScaleKOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // dScaleVOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(dScaleVOptional, dScaleVOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // dScaleDyOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(dScaleDyOptional, dScaleDyOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // dScaleOOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(dScaleOOptional, dScaleOOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    return ret;
}

static void GetInputAndOutputReshapeArray(const aclTensor *query, const aclTensor *key, FagInShapeInfo fagShape,
                                          FagShapeArray &fagShapeArray, aclOpExecutor *executor)
{
    if (fagShape.passThrowInnerFag) {
        return;
    }

    if (fagShape.inputLayoutStr != "BSH" && fagShape.inputLayoutStr != "SBH") {
        return;
    }

    auto queryShape = query->GetViewShape();
    auto keyShape = key->GetViewShape();
    FVector<int64_t, 0> queryReshapeList;
    FVector<int64_t, 0> keyReshapeList;
    FVector<int64_t, 0> dqReshapeList;
    FVector<int64_t, 0> dkReshapeList;
    for (size_t i = 0; i < 3; i++) { // 3: sizeof("BSH")
        dqReshapeList.emplace_back(queryShape.GetDim(i));
        dkReshapeList.emplace_back(keyShape.GetDim(i));
        if (i < 2) { // 2: split last Dim
            queryReshapeList.emplace_back(queryShape.GetDim(i));
            keyReshapeList.emplace_back(keyShape.GetDim(i));
        }
    }

    queryReshapeList.emplace_back(fagShape.n1Dim);
    queryReshapeList.emplace_back(fagShape.dDim);
    keyReshapeList.emplace_back(fagShape.n2Dim);
    keyReshapeList.emplace_back(fagShape.dDim);

    // get shape array
    fagShapeArray.queryShapeArray = executor->AllocIntArray(queryReshapeList.data(), queryReshapeList.size());
    fagShapeArray.dqShapeArray = executor->AllocIntArray(dqReshapeList.data(), dqReshapeList.size());
    fagShapeArray.keyShapeArray = executor->AllocIntArray(keyReshapeList.data(), keyReshapeList.size());
    fagShapeArray.dkShapeArray = executor->AllocIntArray(dkReshapeList.data(), dkReshapeList.size());

    return;
}

static void GetInputAndOutputBackwordReshapeArrayForSBH(const aclTensor *query, const aclTensor *key,
                                                        FagInShapeInfo fagShape, FagShapeArray &fagShapeArray,
                                                        aclOpExecutor *executor)
{
    if (!(fagShape.needBackwordReshape)) {
        return;
    }

    if (query == nullptr || key == nullptr) {
        return;
    }
    auto queryShape = query->GetViewShape();
    auto keyShape = key->GetViewShape();
    FVector<int64_t, 0> queryReshapeList;
    FVector<int64_t, 0> keyReshapeList;
    FVector<int64_t, 0> dqReshapeList;
    FVector<int64_t, 0> dkReshapeList;
    for (size_t i = 0; i < 2; i++) { // 2: get SBH pre shape size 'SB'
        queryReshapeList.emplace_back(queryShape.GetDim(i));
        dqReshapeList.emplace_back(queryShape.GetDim(i));
        keyReshapeList.emplace_back(keyShape.GetDim(i));
        dkReshapeList.emplace_back(keyShape.GetDim(i));
    }

    auto dDimAlignSize = (fagShape.dDim + fagShape.alignDim - 1) / fagShape.alignDim * fagShape.alignDim;
    auto queryHDimAlignSize = fagShape.n1Dim * dDimAlignSize;
    auto keyHDimAlignSize = fagShape.n2Dim * dDimAlignSize;

    queryReshapeList.emplace_back(queryHDimAlignSize);
    keyReshapeList.emplace_back(keyHDimAlignSize);

    dqReshapeList.emplace_back(fagShape.n1Dim);
    dqReshapeList.emplace_back(dDimAlignSize);
    dkReshapeList.emplace_back(fagShape.n2Dim);
    dkReshapeList.emplace_back(dDimAlignSize);

    // get shape array
    fagShapeArray.queryBwShapeArray = executor->AllocIntArray(queryReshapeList.data(), queryReshapeList.size());
    fagShapeArray.dqBwShapeArray = executor->AllocIntArray(dqReshapeList.data(), dqReshapeList.size());
    fagShapeArray.keyBwShapeArray = executor->AllocIntArray(keyReshapeList.data(), keyReshapeList.size());
    fagShapeArray.dkBwShapeArray = executor->AllocIntArray(dkReshapeList.data(), dkReshapeList.size());

    return;
}

static void GetKvUnequalReshapeArray(const aclTensor *value, FagInShapeInfo fagShape, FagShapeArray &fagShapeArray, aclOpExecutor *executor)
{
    if (!(fagShape.needPadValueD)) {
        return;
    }

    if (!(fagShape.inputLayoutStr == "SBH" || fagShape.inputLayoutStr == "BSH")) {
        return;
    }

    FVector<int64_t, DIM_NUM_4> valueReshapeBeforeList;
    FVector<int64_t, DIM_NUM_4> attenInReshapeBeforeList;
    FVector<int64_t, DIM_NUM_4> dvReshapeBeforeList;
    FVector<int64_t, DIM_NUM_3> valueReshapeAfterList;
    FVector<int64_t, DIM_NUM_3> attenInReshapeAfterList;
    FVector<int64_t, DIM_NUM_3> dvReshapeAfterList;

    if (fagShape.inputLayoutStr == "SBH") {
        auto bDim = value->GetViewShape().GetDim(1);
        valueReshapeBeforeList.assign({fagShape.s2Dim, bDim, fagShape.n2Dim, fagShape.dvDim});
        valueReshapeAfterList.assign({fagShape.s2Dim, bDim, fagShape.n2Dim * fagShape.dDim});
        attenInReshapeBeforeList.assign({fagShape.s1Dim, bDim, fagShape.n1Dim, fagShape.dvDim});
        attenInReshapeAfterList.assign({fagShape.s1Dim, bDim, fagShape.n1Dim * fagShape.dDim});
        dvReshapeBeforeList.assign({fagShape.s2Dim, bDim, fagShape.n2Dim, fagShape.dDim});
        dvReshapeAfterList.assign({fagShape.s2Dim, bDim, fagShape.n2Dim * fagShape.dvDim});
    } else { // BSH
        auto bDim = value->GetViewShape().GetDim(0);
        valueReshapeBeforeList.assign({bDim, fagShape.s2Dim, fagShape.n2Dim, fagShape.dvDim});
        valueReshapeAfterList.assign({bDim, fagShape.s2Dim, fagShape.n2Dim * fagShape.dDim});
        attenInReshapeBeforeList.assign({bDim, fagShape.s1Dim, fagShape.n1Dim, fagShape.dvDim});
        attenInReshapeAfterList.assign({bDim, fagShape.s1Dim, fagShape.n1Dim * fagShape.dDim});
        dvReshapeBeforeList.assign({bDim, fagShape.s2Dim, fagShape.n2Dim, fagShape.dDim});
        dvReshapeAfterList.assign({bDim, fagShape.s2Dim, fagShape.n2Dim * fagShape.dvDim});
    }

    fagShapeArray.valueReshapeBefore = executor->AllocIntArray(valueReshapeBeforeList.data(), valueReshapeBeforeList.size());
    fagShapeArray.valueReshapeAfter = executor->AllocIntArray(valueReshapeAfterList.data(), valueReshapeAfterList.size());

    fagShapeArray.attenInReshapeBefore = executor->AllocIntArray(attenInReshapeBeforeList.data(), attenInReshapeBeforeList.size());
    fagShapeArray.attenInReshapeAfter = executor->AllocIntArray(attenInReshapeAfterList.data(), attenInReshapeAfterList.size());

    fagShapeArray.dvReshapeBefore = executor->AllocIntArray(dvReshapeBeforeList.data(), dvReshapeBeforeList.size());
    fagShapeArray.dvReshapeAfter = executor->AllocIntArray(dvReshapeAfterList.data(), dvReshapeAfterList.size());
}

static aclnnStatus ReshapeInputTensor(FagTensorInput &tensorInput,
                                      FagInShapeInfo fagShape, FagShapeArray fagShapeArray, bool isBackWord,
                                      aclOpExecutor *executor)
{
    bool needReshape = isBackWord ? fagShape.needBackwordReshape : !(fagShape.passThrowInnerFag);
    if (!needReshape) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr != "BSH" && fagShape.inputLayoutStr != "SBH") {
        return ACLNN_SUCCESS;
    }

    auto queryShapeArray = isBackWord ? fagShapeArray.queryBwShapeArray : fagShapeArray.queryShapeArray;
    auto keyShapeArray = isBackWord ? fagShapeArray.keyBwShapeArray : fagShapeArray.keyShapeArray;

    // reshape input
    *(tensorInput.queryCngs) = l0op::Reshape(*(tensorInput.queryCngs), queryShapeArray, executor);
    CHECK_RET(*(tensorInput.queryCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *(tensorInput.keyCngs) = l0op::Reshape(*(tensorInput.keyCngs), keyShapeArray, executor);
    CHECK_RET(*(tensorInput.keyCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *(tensorInput.valueCngs) = l0op::Reshape(*(tensorInput.valueCngs), keyShapeArray, executor);
    CHECK_RET(*(tensorInput.valueCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *(tensorInput.dyCngs) = l0op::Reshape(*(tensorInput.dyCngs), queryShapeArray, executor);
    CHECK_RET(*(tensorInput.dyCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (*(tensorInput.attentionInOptionalCngs) != nullptr &&
        (*(tensorInput.attentionInOptionalCngs))->GetViewShape().GetDimNum() != 0) {
        *(tensorInput.attentionInOptionalCngs) = l0op::Reshape(*(tensorInput.attentionInOptionalCngs),
            queryShapeArray, executor);
        CHECK_RET(*(tensorInput.attentionInOptionalCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ReshapeOutputTensor(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                       FagInShapeInfo fagShape, FagShapeArray fagShapeArray, bool isBackWord,
                                       aclOpExecutor *executor)
{
    bool needReshape = isBackWord ? fagShape.needBackwordReshape : !(fagShape.passThrowInnerFag);
    if (!needReshape) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr != "BSH" && fagShape.inputLayoutStr != "SBH") {
        return ACLNN_SUCCESS;
    }

    aclIntArray *dqShapeArray = isBackWord ? fagShapeArray.dqBwShapeArray : fagShapeArray.dqShapeArray;
    aclIntArray *dkShapeArray = isBackWord ? fagShapeArray.dkBwShapeArray : fagShapeArray.dkShapeArray;

    // reshape
    fagOut[DQ_OUT_IDX] = l0op::Reshape(fagOut[DQ_OUT_IDX], dqShapeArray, executor);
    CHECK_RET(fagOut[DQ_OUT_IDX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    fagOut[DK_OUT_IDX] = l0op::Reshape(fagOut[DK_OUT_IDX], dkShapeArray, executor);
    CHECK_RET(fagOut[DK_OUT_IDX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    fagOut[DV_OUT_IDX] = l0op::Reshape(fagOut[DV_OUT_IDX], dkShapeArray, executor); // 2:dv
    CHECK_RET(fagOut[DV_OUT_IDX] != nullptr, ACLNN_ERR_INNER_NULLPTR);     // 2:dv

    return ACLNN_SUCCESS;
}

static aclnnStatus PaddingInputTensorDdim(FagTensorInput &tensorInput,
                                          FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needPadDimD)) {
        OP_LOGD("Fag aclnn case do not do pad dimD operation.");
        return ACLNN_SUCCESS;
    }
    OP_LOGD("Fag aclnn case do pad dimD operation.");

    // padding
    // query
    auto padSize = (fagShape.dDim + fagShape.alignDim - 1) / fagShape.alignDim * fagShape.alignDim - fagShape.dDim;
    aclIntArray *paddingArray = nullptr;
    if (fagShape.inputLayoutStr == "TND") {
        FVector<int64_t> padding = {0, 0, 0, 0, 0, padSize};
        paddingArray = executor->AllocIntArray(padding.data(), 6); // 6: TND 3dims, padding D dim
    } else {
        FVector<int64_t> padding = {0, 0, 0, 0, 0, 0, 0, padSize};
        paddingArray = executor->AllocIntArray(padding.data(), 8); // 8: BNSD 4dims, padding D dim
    }
    auto padTensor = executor->ConvertToTensor(paddingArray, DataType::DT_INT64);
    CHECK_RET(padTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *(tensorInput.queryCngs) = l0op::Pad(*(tensorInput.queryCngs), padTensor, executor);
    CHECK_RET(*(tensorInput.queryCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // key
    *(tensorInput.keyCngs) = l0op::Pad(*(tensorInput.keyCngs), padTensor, executor);
    CHECK_RET(*(tensorInput.keyCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // value
    *(tensorInput.valueCngs) = l0op::Pad(*(tensorInput.valueCngs), padTensor, executor);
    CHECK_RET(*(tensorInput.valueCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // dy
    *(tensorInput.dyCngs) = l0op::Pad(*(tensorInput.dyCngs), padTensor, executor);
    CHECK_RET(*(tensorInput.dyCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // attenmask_in
    if (*(tensorInput.attentionInOptionalCngs) != nullptr &&
        (*(tensorInput.attentionInOptionalCngs))->GetViewShape().GetDimNum() != 0) {
        *(tensorInput.attentionInOptionalCngs) = l0op::Pad(*(tensorInput.attentionInOptionalCngs), padTensor, executor);
        CHECK_RET(*(tensorInput.attentionInOptionalCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus SliceOutputTensorDdim(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                         FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needPadDimD)) {
        return ACLNN_SUCCESS;
    }

    auto dqOutShape = (fagOut[DQ_OUT_IDX])->GetViewShape(); // 0: dq
    auto dkOutShape = (fagOut[DK_OUT_IDX])->GetViewShape(); // 1: dk

    // slice
    FVector<int64_t> dqOutSizeVector;
    FVector<int64_t> dkOutSizeVector;
    for (size_t i = 0; i < dqOutShape.GetDimNum() - 1; i++) {
        dqOutSizeVector.emplace_back(dqOutShape.GetDim(i));
    }

    for (size_t i = 0; i < dkOutShape.GetDimNum() - 1; i++) {
        dkOutSizeVector.emplace_back(dkOutShape.GetDim(i));
    }

    aclIntArray *offsets = nullptr;
    if (fagShape.inputLayoutStr == "TND") {
        FVector<int64_t> offsetsVector = {0, 0, 0};
        offsets = executor->AllocIntArray(offsetsVector.data(), offsetsVector.size());
    } else {
        FVector<int64_t> offsetsVector = {0, 0, 0, 0};
        offsets = executor->AllocIntArray(offsetsVector.data(), offsetsVector.size());
    }

    dqOutSizeVector.emplace_back(fagShape.dDim);
    auto dqOutSize = executor->AllocIntArray(dqOutSizeVector.data(), dqOutSizeVector.size());
    fagOut[DQ_OUT_IDX] = l0op::Slice(fagOut[DQ_OUT_IDX], offsets, dqOutSize, executor); // 0: dq
    CHECK_RET(fagOut[DQ_OUT_IDX] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    dkOutSizeVector.emplace_back(fagShape.dDim);
    auto dkOutSize = executor->AllocIntArray(dkOutSizeVector.data(), dkOutSizeVector.size());
    fagOut[DK_OUT_IDX] = l0op::Slice(fagOut[DK_OUT_IDX], offsets, dkOutSize, executor); // 1: dk
    CHECK_RET(fagOut[DK_OUT_IDX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    fagOut[DV_OUT_IDX] = l0op::Slice(fagOut[DV_OUT_IDX], offsets, dkOutSize, executor); // 2: dv
    CHECK_RET(fagOut[DV_OUT_IDX] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static inline const aclTensor *GeneratePaddings(int32_t dimNum, int32_t padNum, aclOpExecutor *executor)
{
    // 2代表每根轴的前后都可以补0
    FVector<int64_t> padVec(dimNum * 2, 0);
    padVec[padVec.size() - 1] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), padVec.size());
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }

    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    return padTensor;
}

static aclnnStatus TransposeInputTensor(FagTensorInput &tensorInput,
                                        FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needTranspose)) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr == "BNSD" || fagShape.inputLayoutStr == "TND") {
        return ACLNN_SUCCESS;
    }

    FVector<int64_t> transposeDim;
    if (fagShape.inputLayoutStr == "BSH" || fagShape.inputLayoutStr == "BSND") {
        transposeDim = {0, 2, 1, 3};
    } else {
        transposeDim = {1, 2, 0, 3};
    }

    auto perm = executor->AllocIntArray(transposeDim.data(), transposeDim.size());

    // query
    *(tensorInput.queryCngs) = l0op::Transpose(*(tensorInput.queryCngs), perm, executor);
    CHECK_RET(*(tensorInput.queryCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // key
    *(tensorInput.keyCngs) = l0op::Transpose(*(tensorInput.keyCngs), perm, executor);
    CHECK_RET(*(tensorInput.keyCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // value
    *(tensorInput.valueCngs) = l0op::Transpose(*(tensorInput.valueCngs), perm, executor);
    CHECK_RET(*(tensorInput.valueCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // dy
    *(tensorInput.dyCngs) = l0op::Transpose(*(tensorInput.dyCngs), perm, executor);
    CHECK_RET(*(tensorInput.dyCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // attentionInOptional
    if (*(tensorInput.attentionInOptionalCngs) != nullptr &&
        (*(tensorInput.attentionInOptionalCngs))->GetViewShape().GetDimNum() != 0) {
        *(tensorInput.attentionInOptionalCngs) = l0op::Transpose(*(tensorInput.attentionInOptionalCngs), perm, executor);
        CHECK_RET(*(tensorInput.attentionInOptionalCngs) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus TransposeOutputTensor(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                         FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needTranspose)) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr == "BNSD" || fagShape.inputLayoutStr == "TND") {
        return ACLNN_SUCCESS;
    }

    FVector<int64_t> transposeDim;
    if (fagShape.inputLayoutStr == "BSH" || fagShape.inputLayoutStr == "BSND") {
        transposeDim = {0, 2, 1, 3};
    } else {
        transposeDim = {2, 0, 1, 3};
    }

    auto perm = executor->AllocIntArray(transposeDim.data(), transposeDim.size());

    // dqOut
    fagOut[DQ_OUT_IDX] = l0op::Transpose(fagOut[DQ_OUT_IDX], perm, executor);
    CHECK_RET(fagOut[DQ_OUT_IDX] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // dkOut
    fagOut[DK_OUT_IDX] = l0op::Transpose(fagOut[DK_OUT_IDX], perm, executor);
    CHECK_RET(fagOut[DK_OUT_IDX] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // dvOut
    fagOut[DV_OUT_IDX] = l0op::Transpose(fagOut[DV_OUT_IDX], perm, executor);   // 2:dvOut
    CHECK_RET(fagOut[DV_OUT_IDX] != nullptr, ACLNN_ERR_INNER_NULLPTR); // 2:dvOut

    // dpseOut
    return ACLNN_SUCCESS;
}

static aclnnStatus PreFlashAttentionScoreGrad(FagTensorInput &tensorInput,
                                              FagInShapeInfo fagShape, FagShapeArray &fagShapeArray,
                                              aclOpExecutor *executor)
{
    // 获取reshape array, SBH特殊场景下，需要提前获取调用FAG前反向reshape成SBH时所需的reshape array
    GetInputAndOutputReshapeArray(*(tensorInput.queryCngs), *(tensorInput.keyCngs), fagShape, fagShapeArray, executor);
    GetInputAndOutputBackwordReshapeArrayForSBH(*(tensorInput.queryCngs), *(tensorInput.keyCngs),
        fagShape, fagShapeArray, executor);
    GetKvUnequalReshapeArray(*(tensorInput.valueCngs), fagShape, fagShapeArray, executor);

    // 将输入tensor从三维扩展成四维
    auto ret = ReshapeInputTensor(tensorInput, fagShape, fagShapeArray, false, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 执行D轴Padding到对齐值
    ret = PaddingInputTensorDdim(tensorInput, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 执行输入transpose到BNSD
    ret = TransposeInputTensor(tensorInput, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果是SBH特殊场景，在调用FAG前，需要将SBND重新改成SBH，否则FAG将报错不支持layout
    ret = ReshapeInputTensor(tensorInput, fagShape, fagShapeArray, true, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

static aclnnStatus PostFlashAttentionScoreGrad(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                               const aclTensor **dqOut, const aclTensor **dkOut,
                                               const aclTensor **dvOut, const aclTensor **dqRopeOut,
                                               const aclTensor **dkRopeOut, const aclTensor **dpseOut,
                                               FagInShapeInfo fagShape, FagShapeArray &fagShapeArray,
                                               aclOpExecutor *executor)
{
    // 如果是SBH特殊场景，在调用FAG后，需要将SBH重新改成SBND，以完成后续的slice等操作
    auto ret = ReshapeOutputTensor(fagOut, fagShape, fagShapeArray, true, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 将输出由BNSD转为原始shape
    ret = TransposeOutputTensor(fagOut, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 将D轴padding脏数据切掉
    ret = SliceOutputTensorDdim(fagOut, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 将输出tensor由四维还原成三维
    ret = ReshapeOutputTensor(fagOut, fagShape, fagShapeArray, false, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto dqOutViewCopyRes = l0op::ViewCopy(fagOut[DQ_OUT_IDX], *dqOut, executor);
    CHECK_RET(dqOutViewCopyRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dkOutViewCopyRes = l0op::ViewCopy(fagOut[DK_OUT_IDX], *dkOut, executor);
    CHECK_RET(dkOutViewCopyRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dvOutViewCopyRes = l0op::ViewCopy(fagOut[DV_OUT_IDX], *dvOut, executor);
    CHECK_RET(dvOutViewCopyRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (*dpseOut == nullptr || (*dpseOut)->GetDataType() == ge::DataType::DT_FLOAT) {
        return ACLNN_SUCCESS;
    }

    auto dpseOutViewCopyRes = l0op::ViewCopy(fagOut[3], *dpseOut, executor);
    CHECK_RET(dpseOutViewCopyRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (dqRopeOut != nullptr && *dqRopeOut != nullptr) {
        auto dqRopeOutViewCopyRes = l0op::ViewCopy(fagOut[4], *dqRopeOut, executor);
        CHECK_RET(dqRopeOutViewCopyRes != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (dkRopeOut != nullptr && *dkRopeOut != nullptr) {
        auto dkRopeOutViewCopyRes = l0op::ViewCopy(fagOut[5], *dkRopeOut, executor);
        CHECK_RET(dkRopeOutViewCopyRes != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus FlashAttentionScoreGradVXGetWorkspace(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
    const aclTensor *dScaleQOptional, const aclTensor *dScaleKOptional, const aclTensor *dScaleVOptional,
    const aclTensor *dScaleDyOptional, const aclTensor *dScaleOOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t outDtypeOptional, int64_t pseType,
    int64_t seed, int64_t offset, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dqRopeOut, const aclTensor *dkRopeOut,
    const aclTensor *dpseOut, aclOpExecutor *executor) {
    // 获取基本参数
    FagInShapeInfo fagShape;
    FagShapeCollection shapeCollect = {query, key, value, dy, attentionInOptional,
        headNum, inputLayout, &fagShape, actualSeqQLenOptional, actualSeqKvLenOptional, keepProb};
    // 检查tensor维度是否大于2
    auto ret = InvalidTensorDimCheck(shapeCollect, dqOut, dkOut, dvOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = GetInputShapeInfo(shapeCollect);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, pls check input shape.");
        return ret;
    }

    // 输入连续性转换
    const aclTensor *queryCngs = nullptr;
    const aclTensor *keyCngs = nullptr;
    const aclTensor *valueCngs = nullptr;
    const aclTensor *dyCngs = nullptr;
    const aclTensor *attentionInOptionalCngs = nullptr;
    const aclTensor *pseShiftOptionalCngs = nullptr;
    const aclTensor *dropMaskOptionalCngs = nullptr;
    const aclTensor *paddingMaskOptionalCngs = nullptr;
    const aclTensor *attenMaskOptionalCngs = nullptr;
    const aclTensor *softmaxMaxOptionalCngs = nullptr;
    const aclTensor *softmaxSumOptionalCngs = nullptr;
    const aclTensor *softmaxInOptionalCngs = nullptr;
    const aclTensor *queryRopeOptionalCngs = nullptr;
    const aclTensor *keyRopeOptionalCngs = nullptr;
    const aclTensor *dScaleQOptionalCngs = nullptr;
    const aclTensor *dScaleKOptionalCngs = nullptr;
    const aclTensor *dScaleVOptionalCngs = nullptr;
    const aclTensor *dScaleDyOptionalCngs = nullptr;
    const aclTensor *dScaleOOptionalCngs = nullptr;
    FagTensorInput tensorInput = {query, key, value, dy, attentionInOptional, &queryCngs, &keyCngs, &valueCngs, &dyCngs,
                                &attentionInOptionalCngs};
    ret = ContiguousInputTensor(tensorInput, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ContiguousOptionalInputTensor(
        pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional,
        softmaxSumOptional, softmaxInOptional, queryRopeOptional, keyRopeOptional,
        dScaleQOptional, dScaleKOptional, dScaleVOptional, dScaleDyOptional, dScaleOOptional,
        &pseShiftOptionalCngs, &dropMaskOptionalCngs, &paddingMaskOptionalCngs,
        &attenMaskOptionalCngs, &softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, &softmaxInOptionalCngs,
        &queryRopeOptionalCngs, &keyRopeOptionalCngs, &dScaleQOptionalCngs, &dScaleKOptionalCngs,
        &dScaleVOptionalCngs, &dScaleDyOptionalCngs, &dScaleOOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // reshape + PAD + Transpose
    FagShapeArray fagShapeArray;
    ret = PreFlashAttentionScoreGrad(tensorInput, fagShape,
                                     fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 调整input layout
    char inputLayoutUnderTrans[MAX_LAYOUT_SIZE] = {0};
    ConvertInputLayout(fagShape, inputLayout, inputLayoutUnderTrans, MAX_LAYOUT_SIZE);

    // 调用FAG ascendc接口
    auto fagRes = l0op::FlashAttentionScoreGrad(
        queryCngs, keyCngs, valueCngs, dyCngs, pseShiftOptionalCngs, dropMaskOptionalCngs, paddingMaskOptionalCngs,
        attenMaskOptionalCngs, softmaxMaxOptionalCngs, softmaxSumOptionalCngs, softmaxInOptionalCngs,
        attentionInOptionalCngs, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
        kvStartIdxOptional, dScaleQOptionalCngs, dScaleKOptionalCngs, dScaleVOptionalCngs, dScaleDyOptionalCngs,
        dScaleOOptionalCngs, queryRopeOptionalCngs, keyRopeOptionalCngs, scaleValue, keepProb, preTokens, nextTokens,
        headNum, inputLayoutUnderTrans, innerPrecise, sparseMode, pseType, seed, offset,  outDtypeOptional, defaultSoftmaxInLayoutRegbase, executor);
    CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr,  // 0: dqOut 1: dkOut 2:dvOut
              ACLNN_ERR_INNER_NULLPTR);

    // transpose + slice + reshape + viewCopy
    ret = PostFlashAttentionScoreGrad(fagRes, &dqOut, &dkOut, &dvOut, &dqRopeOut, &dkRopeOut, &dpseOut, fagShape,
                                      fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradVXGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *dScaleQOptional,
    const aclTensor *dScaleKOptional, const aclTensor *dScaleVOptional, const aclTensor *dScaleDyOptional,
    const aclTensor *dScaleOOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValueOptional,
    double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum,
    char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional, int64_t pseTypeOptional,
    int64_t seed, int64_t offset, int64_t outDtypeOptional,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dqRopeOut, const aclTensor *dkRopeOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor) 
{
    L2_DFX_PHASE_1(aclnnFlashAttentionScoreGradVX,
        DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
               softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, queryRopeOptional,
               keyRopeOptional, dScaleQOptional, dScaleKOptional, dScaleVOptional, dScaleDyOptional, dScaleOOptional,
               prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional,
               scaleValueOptional, keepProbOptional, preTokensOptional, nextTokensOptional, headNum, inputLayout,
               innerPreciseOptional, sparseModeOptional, pseTypeOptional, seed, offset, outDtypeOptional),
        DFX_OUT(dqOut, dkOut, dvOut, dqRopeOut, dkRopeOut, dpseOut));
 
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
 
    // 空Tensor处理
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(keyIn != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(attentionInOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dqOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dkOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dvOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (dqOut->IsEmpty() && dkOut->IsEmpty() && dvOut->IsEmpty()) {
        if (dpseOut == nullptr || dpseOut->IsEmpty()) {
            OP_LOGD("All out tensor is empty");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }
 
    // 异常防护
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid HeadNum, pls check input attr");
        return ACLNN_ERR_PARAM_INVALID;
    }
 
    // calculate fag
    auto ret = FlashAttentionScoreGradVXGetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, queryRopeOptional,
        keyRopeOptional, dScaleQOptional, dScaleKOptional, dScaleVOptional, dScaleDyOptional, dScaleOOptional,
        prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional,
        scaleValueOptional, keepProbOptional, preTokensOptional, nextTokensOptional, headNum, inputLayout,
        innerPreciseOptional, sparseModeOptional, outDtypeOptional, pseTypeOptional, seed, offset, dqOut, dkOut,
        dvOut, dqRopeOut, dkRopeOut, dpseOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
 
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
 
aclnnStatus aclnnFlashAttentionScoreGradVX(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    const aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreGradVX);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}  // namespace

#ifdef __cplusplus
}
#endif
