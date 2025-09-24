/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _MOE_TOKEN_PERMUTE_GRAD_TILING_H_
#define _MOE_TOKEN_PERMUTE_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct MoeTokenPermuteGradTilingData {
  int64_t hidden_size = 0;
  int64_t top_k = 0;
  int64_t num_out_tokens = 0;
  int64_t hidden_splited_length = 0;
  int64_t hidden_splited_num = 0;
  int64_t hidden_splited_remain = 0;
  int64_t tokens_core_length = 0;
  int64_t tokens_core_remain = 0;
  int64_t tokens_splited_length = 0;
  int64_t tokens_splited_num = 0;
  int64_t tokens_splited_remain = 0;
  int64_t buffer_num = 0;
};

inline void InitMoeTokenPermuteGradTilingData(uint8_t* tiling, MoeTokenPermuteGradTilingData* const_data) {
  memcpy(const_data, tiling, sizeof(MoeTokenPermuteGradTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
  MoeTokenPermuteGradTilingData tilingData;          \
  InitMoeTokenPermuteGradTilingData(tilingPointer, &tilingData)
#endif