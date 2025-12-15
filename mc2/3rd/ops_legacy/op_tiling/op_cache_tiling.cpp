/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file op_cache_tiling.cpp
 * \brief
 */

#include "op_cache_tiling.h"
#include "legacy_common_manager.h"
#include "log/log.h"
namespace optiling {

template<typename FuncType>
FuncType LoadLegacyFunc(const char* symbolName, const char* funcDesc) {
    static FuncType func = []() -> FuncType {
        auto& mgr = Ops::MC2::LegacyCommonMgr::GetInstance();
        auto funcPtr = mgr.GetFunc<FuncType>(symbolName);
        if (funcPtr == nullptr) {
            OP_LOGE("LegacyCommonMgr", "FATAL: Load legacy func [%s] (symbol: %s) failed! Symbol not found.",
                    funcDesc, symbolName);
        } else {
            OP_LOGI("LegacyCommonMgr", "Load legacy func [%s] (symbol: %s) success.",
                    funcDesc, symbolName);
        }
        return funcPtr;
    }();
    return func;
}

bool TilingPrepareForOpCache(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForOpCache", "Input TilingContext is null! Return false.");
        return false;
    }
    using FuncType = bool (*)(gert::TilingContext *);
    auto func = LoadLegacyFunc<FuncType>("LegacyTilingPrepareForOpCache", "TilingPrepareForOpCache");
    if (func == nullptr) {
        return false;
    }
    return func(context);
}

bool TilingPrepareForOpCache(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForOpCache", "Input TilingContext is null! Return false.");
        return false;
    }
    using FuncType = bool (*)(gert::TilingParseContext *);
    auto func = LoadLegacyFunc<FuncType>("LegacyTilingParsePrepareForOpCache", "TilingPrepareForOpCache");
    if (func == nullptr) {
        return false;
    }
    return func(context);
}

bool GenTiling(const std::string &op_type, const BatchmatmulCompileParas &compile_params,
               BatchmatmulRunParas &run_params, CacheTilingData &tiling, gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE("GenTiling", "Input TilingContext is null! Return false.");
        return false;
    }
    using FuncType = bool (*)(const std::string &, const BatchmatmulCompileParas &, BatchmatmulRunParas &,
                              CacheTilingData &, gert::TilingContext *);
    auto func = LoadLegacyFunc<FuncType>("LegacyGenTbeMatmulTiling", "GenTiling");
    if (func == nullptr) {
        return false;
    }
    return func(op_type, compile_params, run_params, tiling, context);
}

bool CheckSupportConditionQbmm(QbmmType type, QuantBatchMatmulRunParas &inputParams, uint64_t aicNum,
                               bool supportL0c2Out)
{
    using FuncType = bool (*)(QbmmType, QuantBatchMatmulRunParas &, uint64_t, bool);
    auto func = LoadLegacyFunc<FuncType>("LegacyCheckSupportConditionQbmm", "CheckSupportConditionQbmm");
    if (func == nullptr) {
        return false;
    }
    return func(type, inputParams, aicNum, supportL0c2Out);
}

bool GenWqbmmTiling(const std::string &op_type, const WeightQuantBatchMatmulCacheTilingParas &compile_params,
                    WeightQuantBatchMatmulCacheTilingData &cacheTiling)
{
    using FuncType = bool (*)(const std::string &, const WeightQuantBatchMatmulCacheTilingParas &,
                              WeightQuantBatchMatmulCacheTilingData &);
    auto func = LoadLegacyFunc<FuncType>("LegacyGenWqbmmTiling", "GenWqbmmTiling");
    if (func == nullptr) {
        return false;
    }
    return func(op_type, compile_params, cacheTiling);
}
} // namespace optiling