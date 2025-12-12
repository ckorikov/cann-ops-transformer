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
 * \file quant_all_reduce_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef QUANT_ALL_REDUCE_TILING_DATA_H
#define QUANT_ALL_REDUCE_TILING_DATA_H

#include <kernel_tiling/kernel_tiling.h>

struct QuantAllReduceTilingInfo {
    uint64_t bs;              // bs轴
    uint64_t hiddenSize;      // x的h轴
    uint64_t scaleHiddenSize; // scales的h轴
    uint64_t aivNum;          // aiv数
};

struct QuantAllReduceTilingData {
    Mc2InitTiling mc2InitTiling; // 初始化通信任务配置
    Mc2CcTiling mc2CcTiling;     // 具体每个通信任务的参数配置
    QuantAllReduceTilingInfo quantAllReduceTilingInfo;
};

#endif // QUANT_ALL_REDUCE_TILING_DATA_H