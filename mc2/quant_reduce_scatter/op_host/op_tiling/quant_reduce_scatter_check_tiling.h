/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QUANT_REDECE_SCATTER_CHECK_TILING_H
#define QUANT_REDECE_SCATTER_CHECK_TILING_H
#include "tiling/mc2_tiling_utils.h"
namespace MC2Tiling {
// tiling函数使用的中间变量
struct QuantReduceScatterTilingParams {
    uint32_t quantMode;   // 量化方式
    uint32_t rankSize;
};

class QuantReduceScatterCheckTiling {
public:
    static ge::graphStatus CheckAttrs(const gert::TilingContext *context);
    static bool CheckDimWithQuantMode(const gert::TilingContext *context,
                                      const QuantReduceScatterTilingParams &params);
    static bool CheckInputTensorDim(const gert::TilingContext *context, const QuantReduceScatterTilingParams &params);
    static bool CheckOutputTensorDim(const gert::TilingContext *context, const QuantReduceScatterTilingParams &params);
    static bool CheckInputDtypeAndSetQuantMode(const gert::TilingContext *context,
                                               QuantReduceScatterTilingParams &params);
    static bool CheckTensorDataType(const gert::TilingContext *context, QuantReduceScatterTilingParams &params);
    static bool CheckTensorDim(const gert::TilingContext *context, const QuantReduceScatterTilingParams &params);
    static bool CheckTensorFormat(const gert::TilingContext *context);
    static bool CheckWindowSize(const gert::TilingContext *context, const QuantReduceScatterTilingParams &params);
    static ge::graphStatus TilingCheckQuantReduceScatter(const gert::TilingContext *context,
                                                         QuantReduceScatterTilingParams &params);
};
}; // namespace MC2Tiling
#endif //__QUANT_REDECE_SCATTER_TILING_CHECK_H__