/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file flash_attention_score_common.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_COMMON_REGBASE_H
#define FLASH_ATTENTION_SCORE_COMMON_REGBASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stdarg.h"
#include "pse.h"

using matmul::MatmulType;
using namespace AscendC;

__aicore__ inline constexpr MatmulConfig GetFACustomCfg(bool enableSetTail = true,
    const IterateMode iterateMode = IterateMode::ITERATE_MODE_DEFAULT, bool isC1Shared = false,
    uint32_t sharedC1BufferSize = 64 * 1024, uint32_t singleM = 0, uint32_t singleN = 0, uint32_t singleK = 0,
    uint32_t baseM = 0, uint32_t baseN = 0, uint32_t baseK = 0, bool enableSetDefineData = false,
    bool isA2B2Shared = false)
{
    MatmulShapeParams shapeParams = {singleM, singleN, singleK, baseM, baseN, baseK};
    auto mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);

    mmCfg.intrinsicsCheck = false;
    mmCfg.enUnitFlag = false;
    mmCfg.enableInit = false;
    mmCfg.enableSetBias = false;
    mmCfg.enableQuantVector = false;
    mmCfg.isBiasBatch = false;

    mmCfg.isCO1Shared = isC1Shared;
    mmCfg.iterateMode = iterateMode;
    mmCfg.enableSetTail = enableSetTail;
    mmCfg.enableSetDefineData = enableSetDefineData;
    mmCfg.isA2B2Shared = isA2B2Shared;
    mmCfg.sharedCO1BufferSize = sharedC1BufferSize;
    return mmCfg;
}

constexpr uint64_t BLOCK_BYTE = 32;
constexpr int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
constexpr int32_t SOFTMAX_K_ALIGNED_SIZE = 64;
constexpr uint64_t DATACOPYPAD_PADDING_VALUE_ZERO = 0;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP16 = 0xFBFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP32 = 0x7F7FFFFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP16 = 0x7BFF;
constexpr int64_t pse1NS1S2 = 2;
constexpr int64_t FP8_QUANT_BLOCK_SIZE = 128;
// 0级接口的block间隔范围需要满足32B对齐
constexpr int64_t attenMaskBN2GS1S2 = 0;
constexpr int64_t attenMaskBS1S2 = 1;
constexpr int64_t attenMaskS1S2 = 2;
constexpr int64_t attenMaskTT = 99;
constexpr uint16_t PREFIX_N_MAX_B = 32;
constexpr int32_t fp32BaseSize = 8;

constexpr uint32_t attenMaskNoCompress = 0;
constexpr uint32_t attenMaskLeftUpCausalCompress = 1;
constexpr uint32_t attenMaskRightDownCausalCompress = 2;
constexpr uint16_t SHIFT_NUM_2 = 2;
constexpr uint16_t SHIFT_NUM_6 = 6;
constexpr uint16_t ADD_NUM_63 = 63;

constexpr uint32_t L0C_SHARED_SIZE_64K = 64 * 1024;
constexpr uint32_t L0C_SHARED_SIZE_128K = 128 * 1024;
constexpr uint32_t CV_RATIO = 2;
constexpr uint64_t SYNC_MODE = 4;
constexpr uint64_t MM2_RES_INTRA_EVENT[2] = {7, 8}; // mm2ResIntraEvent
constexpr uint64_t MM1_RES_INTRA_EVENT[2] = {9, 10}; //mm1ResIntraEvent
enum class SparseModeEnum {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    BAND_COMPRESS = 6,
    RIGHT_DOWN_CAUSAL = 7,
    RIGHT_DOWN_CAUSAL_BAND = 8,
    BAND_LEFT_UP_CAUSAL = 9,
};

enum class ImplModeEnum {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

namespace BaseApi {
template <typename INPUT_T>
__aicore__ constexpr bool IsFp8OnlyWithAttenMask(
    regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop) {
    if constexpr (!IsSameType<INPUT_T, fp8_e5m2_t>::value &&
                  !IsSameType<INPUT_T, fp8_e4m3fn_t>::value &&
                  !IsSameType<INPUT_T, hifloat8_t>::value) {
        return false;
    }
    if (pseMode == regbaseutil::PseTypeEnum::PSE_NONE_TYPE && hasAtten && !hasDrop) {
        return true;
    }
    return false;
}

__aicore__ constexpr bool ContainOptionalInput(
    regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop) {
    if (pseMode == regbaseutil::PseTypeEnum::PSE_NONE_TYPE && !hasAtten && !hasDrop) {
        return false;
    } else {
        return true;
    }
}

__aicore__ constexpr bool IsDn(
    bool isFp32, regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool isS1Base64,
    regbaseutil::DTemplateType dTemplateType, bool hasRope) {
    if (!isFp32 && !ContainOptionalInput(pseMode, hasAtten, hasDrop) && !isS1Base64 &&
        (uint16_t)dTemplateType <= (uint16_t)regbaseutil::DTemplateType::Aligned256 && !hasRope) {
        return true;
    }
    return false;
}

template <typename INPUT_T>
__aicore__ constexpr bool UbOutCondition(
    bool isFp32, regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool isS2Base64) {
    if (IsFp8OnlyWithAttenMask<INPUT_T>(pseMode, hasAtten, hasDrop)) {
        return true;
    }
    if (!ContainOptionalInput(pseMode, hasAtten, hasDrop)) {
        if (!isS2Base64 || isFp32) {
            return true;
        }
    }
    return false;
}

__aicore__ constexpr TPosition GetC2Position(regbaseutil::DTemplateType dTemplateType, bool ubOutCondition, bool isNdS2Size256) {
    if ((uint16_t)dTemplateType <= (uint16_t)regbaseutil::DTemplateType::Aligned128 ||
        (ubOutCondition && (uint16_t)dTemplateType <= (uint16_t)regbaseutil::DTemplateType::Aligned192) ||
        isNdS2Size256) {
        return TPosition::VECCALC;
    } else {
        return TPosition::GM;
    }
}
}
#endif // FLASH_ATTENTION_SCORE_COMMON_REGBASE_H