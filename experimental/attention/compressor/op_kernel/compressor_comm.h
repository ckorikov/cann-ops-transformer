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

enum class X_LAYOUT : std::uint8_t {
    BSH = static_cast<std::uint8_t>(0),
    TH = static_cast<std::uint8_t>(1)
};

enum class X_DTYPE : std::uint8_t {
    BF16 = static_cast<std::uint8_t>(0),
    FP16 = static_cast<std::uint8_t>(1)
};

enum class COFF : std::uint8_t {
    DISABLE = static_cast<std::uint8_t>(0),
    OVERLAP = static_cast<std::uint8_t>(1)
};

template <X_LAYOUT X_L, X_DTYPE X_T, COFF C, bool ROTARY_MODE, typename... Args>
struct COMPType {
    static constexpr X_LAYOUT xLayout = X_L;
    static constexpr X_DTYPE xDtype = X_T;
    static constexpr COFF coff = C;
    static constexpr bool rotaryMode = ROTARY_MODE;
};

}
#endif