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
 * \file sparse_flash_attention_antiquant_metadata.h
 * \brief
 */

#ifndef KVQUANT_SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_H
#define KVQUANT_SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_H

#include <cstdint>

namespace optiling {
static constexpr uint32_t AIC_CORE_NUM = 36;  //TODO 根据编译宏确定 aicpu与kernel的宏保持一致
static constexpr uint32_t MAX_FD_NUM = 36;
constexpr uint32_t SCFA_META_SIZE = AIC_CORE_NUM;
using SCFA_METADATA_T = int32_t;

namespace detail {
    // 分核功能模块输出：FD信息，包含需要归约的数据索引及其分核信息
    struct FlashDecodeResult {
        uint32_t fdNum = 0U;
        uint32_t fdBN2Idx[MAX_FD_NUM];
        uint32_t fdMIdx[MAX_FD_NUM];
        uint32_t fdS2SplitNum[MAX_FD_NUM];
    };
    struct SasMetaData{
        uint32_t usedCoreNum = 0U;
        uint32_t mBaseSize = 0U;
        uint32_t s2BaseSize = 0U;
        uint32_t bN2End[AIC_CORE_NUM];
        uint32_t mEnd[AIC_CORE_NUM];
        uint32_t s2End[AIC_CORE_NUM];
        uint32_t headFdDataIdx[AIC_CORE_NUM]; // 每个core处理的第1个归约任务的数据应存放的workspace位置
        struct FlashDecodeResult fdRes;
    };
};
// static_assert(SCFA_META_SIZE * sizeof(SCFA_METADATA_T) >= sizeof(detail::SasMetaData));
};

#endif
