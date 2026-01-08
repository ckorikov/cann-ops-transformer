/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_comm.h
 * \brief
 */

#ifndef COMPRESSOR_COMM_H
#define COMPRESSOR_COMM_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;

namespace Compressor {

enum class QUANT_MODE : std::uint8_t {
    NO_QUANT = static_cast<std::uint8_t>(0),
    QUANT = static_cast<std::uint8_t>(1)
};

enum class ROTARY_MODE : std::uint8_t {
    HALF = static_cast<std::uint8_t>(0),
    INTERLEAVE = static_cast<std::uint8_t>(1)
};



}
#endif