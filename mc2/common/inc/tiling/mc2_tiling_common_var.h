/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file mc2_tiling_common_var.h
 * \brief
 */

#ifndef __MC2_TILING_COMMON_VAR_H__
#define __MC2_TILING_COMMON_VAR_H__

#include "tiling/tiling_type.h"

namespace optiling {
constexpr int64_t MAX_HCCL_HANDLE_LIMIT = 32;
constexpr uint32_t FP32_DATASIZE = 4;
constexpr uint32_t FP16_DATASIZE = 2;
constexpr uint32_t ALIGN32 = 32;
constexpr uint32_t ALIGN16 = 16;
constexpr uint32_t ARR_LENGTH = 128;
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4;  // 只通信不计算
}  // namespace optiling

#endif