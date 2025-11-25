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
 * \file mla_preprocess_tiling.h
 * \brief
 */
#ifndef OPTILING_PARAMS_PREPROCESS_TILING
#define OPTILING_PARAMS_PREPROCESS_TILING

#include <cstdint>
#include <string>
#include <sstream>

namespace optiling {
namespace OpParam {
struct MlaPreprocessParam {
    enum class QuantMode : uint64_t {
        PER_TENSOR_ASYMM_QUANT = 0,
        PER_TOKEN_SYMM_QUANT,
        PER_TOKEN_ASYMM_QUANT,
        NO_QUANT,
    };
    uint64_t N = 128;
    uint64_t headNum = 0;
    uint64_t cacheMode = 0;
    QuantMode quantMode = QuantMode::PER_TENSOR_ASYMM_QUANT;
    bool operator==(const MlaPreprocessParam &other) const
    {
        return N == other.N && headNum == other.headNum && cacheMode == other.cacheMode && quantMode == other.quantMode;
    }
};
} // namespace OpParam
} // namespace optiling

#endif // OPTILING_PARAMS_MLA_PRE_H