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
 * \file chunk_gated_delta_rule_inverse_utils.h
 * \brief
 */

#ifndef __GATED_DELTA_RULE_INVERSE_KERNEL_UTILS_H_
#define __GATED_DELTA_RULE_INVERSE_KERNEL_UTILS_H_

#include "kernel_operator.h"


namespace LowerTriangularInverse {

using namespace AscendC;

struct InitParams {
  GM_ADDR x;
  GM_ADDR y;
  GM_ADDR workspace;
  TPipe *tPipeIn;
  LowerTriangularInverseTilingData *tilingData;
};

constexpr uint32_t BASE_BLOCK_BYTE = 32; // 分形中的元素个数固定为 A:16*(32B/sizof(T)), B:(32B/sizof(T))*16, C:16*16 
constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t MAX_LEN = 128;
constexpr uint32_t BUFFER_NUM = 2;



} // namespace LowerTriangularInverse
#endif // __GATED_DELTA_RULE_INVERSE_KERNEL_UTILS_H_