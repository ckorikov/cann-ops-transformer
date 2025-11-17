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
 * \file all_gather_add_tiling.h
 * \brief
 */

#ifndef __ALL_GATHER_ADD_TILING_H__
#define __ALL_GATHER_ADD_TILING_H__

#include "kernel_tiling/kernel_tiling.h"

struct AllGatherAddTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    uint32_t commTurn; // 通信轮次，本示例通信1次
    uint32_t tileNum; // 每个核需要参与Add计算的数据块个数
    uint32_t totalLength; // 需要参与Add计算的数据总个数
    uint32_t blockLength; // 每个核需要计算的数据个数
    uint32_t tileLength; // 每个核内每个数据块的数据个数
    uint32_t gatherTileLength; // 参与AllGather的数据个数
};

#endif //__ALL_GATHER_ADD_TILING_H__