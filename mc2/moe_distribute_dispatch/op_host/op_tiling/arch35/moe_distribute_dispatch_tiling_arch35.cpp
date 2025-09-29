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
 * \file moe_distribute_dispatch_tiling_arch35.cpp
 * \brief
 */

#include "moe_distribute_dispatch_tiling_arch35.h"

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/mc2_tiling_utils.h"

namespace {
constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_GROUP_TP_INDEX = 4;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 10;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 11;
constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 12;
constexpr uint32_t ATTR_Y_DATATYPE_INDEX = 13;

const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MAX_EP_WORLD_SIZE = 288;
const int64_t MAX_TP_WORLD_SIZE = 2;
const int64_t BS_UPPER_BOUND = 512;

constexpr uint32_t HCCL_CMD_ALLGATHER = 6U;
constexpr uint32_t HCCL_CMD_ALLTOALLV = 8U;
constexpr uint32_t HCCL_VERSION = 3U;

const uint64_t TILING_KEY_BASE_A5 = 1000000000000000000;
constexpr uint32_t NUM_0 = 0;
constexpr uint32_t NUM_1 = 1;
constexpr uint32_t NUM_10 = 10;
constexpr uint32_t NUM_100 = 100;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
constexpr int64_t K_MAX = 8;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;

constexpr uint32_t DAVID_EP_WORLD_SIZE_FOUR = 4;
constexpr uint32_t DAVID_EP_WORLD_SIZE_TWO = 2;

constexpr uint64_t MX_BLOCK_SIZE = 32U;
constexpr uint64_t PERTILE_BLOCK_SIZE = 128U;

constexpr uint64_t STATIC_SCALE_DIM_0 = 1;
constexpr uint64_t HIF8_SCALE_DIM_0 = 1;
constexpr uint64_t ONE_DIM_SCALE_COL_NUM = 1;

constexpr uint32_t MAX_UINT32 = 4294967295;

const std::string OP_NAME = "MoeDistributeDispatchA5";
}

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, MoeDistributeDispatchTilingDataA5 &tilingData)
{
    OP_LOGD(nodeName, "version %u", tilingData.get_version());
    OP_LOGD(nodeName, "hcommCnt %u", tilingData.get_hcommCnt());
    
    OP_LOGD(nodeName, "srcDataType %u", tilingData.hcommCfgATA.get_srcDataType());
    OP_LOGD(nodeName, "dstDataType %u", tilingData.hcommCfgATA.get_dstDataType());
    OP_LOGD(nodeName, "opType %u", tilingData.hcommCfgATA.get_opType());

    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.dispatchTilingInfo.get_epWorldSize());
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.dispatchTilingInfo.get_tpWorldSize());
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.dispatchTilingInfo.get_epRankId());
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.dispatchTilingInfo.get_tpRankId());
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.dispatchTilingInfo.get_expertShardType());
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.dispatchTilingInfo.get_sharedExpertRankNum());
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.dispatchTilingInfo.get_moeExpertNum());
    OP_LOGD(nodeName, "quantMode is %u.", tilingData.dispatchTilingInfo.get_quantMode());
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.dispatchTilingInfo.get_globalBs());
    OP_LOGD(nodeName, "isQuant is %d.", tilingData.dispatchTilingInfo.get_isQuant());
    OP_LOGD(nodeName, "bs is %u.", tilingData.dispatchTilingInfo.get_bs());
    OP_LOGD(nodeName, "k is %u.", tilingData.dispatchTilingInfo.get_k());
    OP_LOGD(nodeName, "h is %u.", tilingData.dispatchTilingInfo.get_h());
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.dispatchTilingInfo.get_aivNum());
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.dispatchTilingInfo.get_totalUbSize());
    OP_LOGD(nodeName, "totalWinSize is %lu.", tilingData.dispatchTilingInfo.get_totalWinSize());
    OP_LOGD(nodeName, "expertTokenNumsType is %u.", tilingData.dispatchTilingInfo.get_expertTokenNumsType());

    OP_LOGD(nodeName, "scalesCol is %lu.", tilingData.dispatchTilingInfo.get_scalesCol());
    OP_LOGD(nodeName, "scalesRow is %lu.", tilingData.dispatchTilingInfo.get_scalesRow());
    OP_LOGD(nodeName, "scalesTypeSize is %u.", tilingData.dispatchTilingInfo.get_scalesTypeSize());
    OP_LOGD(nodeName, "scalesCount is %lu.", tilingData.dispatchTilingInfo.get_scalesCount());
}

inline ge::graphStatus CheckTpRankAttrs(const char *nodeName, const int64_t *tpWorldSizePtr,
    const int64_t *tpRankIdPtr, const char *groupTpPtr)
{
    if (*tpWorldSizePtr > 1) {
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OP_LOGE(nodeName, "The valid range of tpRankId is [0, %ld), but actually got tpRankId=%ld.",
            *tpWorldSizePtr, *tpRankIdPtr), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr), OP_LOGE(nodeName, "The groupTpPtr is null."), return ge::GRAPH_FAILED);
        uint64_t len = strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH);
        OP_TILING_CHECK((len == 0) || (len == MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "Valid length of groupTp must be in the range (0, %lu), but got strnlen(groupTp)=%lu.", 
            MAX_GROUP_NAME_LENGTH, len), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(*tpRankIdPtr != 0,
            OP_LOGE(nodeName, "The expected value of tpRankId is 0 in NoTp mode, but the actual value is %ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}


inline ge::graphStatus CheckEpAndTpWorldAttrs(const char *nodeName, const int64_t *epWorldSizePtr, 
    const int64_t *tpWorldSizePtr, const int64_t *epRankIdPtr, const char *groupEpPtr)
{
    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE(nodeName, "The groupEpPtr is null."), return ge::GRAPH_FAILED);
    uint64_t len = strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH);
    OP_TILING_CHECK((len == 0) || (len == MAX_GROUP_NAME_LENGTH),
        OP_LOGE(nodeName, "Valid length of groupEp must be in the range (0, %lu), but actually got strnlen(groupEp)=%lu.",
            MAX_GROUP_NAME_LENGTH, len), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epWorldSizePtr <= 0) || (*epWorldSizePtr > MAX_EP_WORLD_SIZE),
        OP_LOGE(nodeName, "The valid range of epWorldSize is (0, %ld], but actually got epWorldSize=%ld.",
        MAX_EP_WORLD_SIZE, *epWorldSizePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
        OP_LOGE(nodeName, "The valid range of tpWorldSize is [0, %ld], but actually got tpWorldSize=%ld.",
        MAX_TP_WORLD_SIZE, *tpWorldSizePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr),
        OP_LOGE(nodeName, "The valid range of epRankId is [0, %ld), but actually got epRankId=%ld.",
        *epWorldSizePtr, *epRankIdPtr), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckQuantAndExpertAttrs(const char *nodeName, const int64_t *sharedExpertNumPtr,
    const int64_t quantMode, const int64_t *expertTokenNumsTypePtr)
{
    OP_TILING_CHECK((quantMode < static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
        (quantMode >= static_cast<int64_t>(QuantModeA5::BUTT)),
        OP_LOGE(nodeName, "The valid range of quantMode is [0, %ld), but actually got quantMode=%ld.",
        static_cast<int64_t>(QuantModeA5::BUTT), quantMode), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*sharedExpertNumPtr != 1, OP_LOGE(nodeName,
        "The expected value of sharedExpertNum is 1, but the actual value is %ld.", *sharedExpertNumPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*expertTokenNumsTypePtr != 0) && (*expertTokenNumsTypePtr != 1), 
        OP_LOGE(nodeName, "The expected value of expertTokenNumsType is 0 or 1, but the actual value is %ld.",
        *expertTokenNumsTypePtr), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckExpertAttrs(const char *nodeName, const int64_t *expertShardPtr, 
    const int64_t *sharedExpertRankNumPtr, const int64_t *moeExpertNumPtr, const int64_t *epWorldSizePtr)
{
    OP_TILING_CHECK(*expertShardPtr != 0,
        OP_LOGE(nodeName, "The expected value of expertShardType is 0, but the actual value is %ld.",
        *expertShardPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sharedExpertRankNumPtr < 0) || (*sharedExpertRankNumPtr >= *epWorldSizePtr),
        OP_LOGE(nodeName, "The valid range of sharedExpertRankNum is [0, %ld), but actually got sharedExpertRankNum=%ld.",
        *epWorldSizePtr, *sharedExpertRankNumPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MOE_EXPERT_MAX_NUM),
        OP_LOGE(nodeName, "The valid range of moeExpertNum is (0, %ld], but actually got moeExpertNum=%ld.",
        MOE_EXPERT_MAX_NUM, *moeExpertNumPtr), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOutputDataType(const gert::TilingContext *context, const char *nodeName, const int64_t quantMode)
{
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "Failed to get expandX datatype."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((quantMode == static_cast<int64_t>(QuantModeA5::NON_QUANT)) 
        && (NON_QUANT_DTYPE.find(static_cast<ge::DataType>(expandXDesc->GetDataType())) == NON_QUANT_DTYPE.end()), 
        OP_LOGE(nodeName, 
        "Invalid expandX datatype for quantMode %ld. Only bf16/fp16/hif8/fp8_e4m3fn/fp8_e5m2 is supported, but got %s.", 
        quantMode, Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((quantMode == static_cast<int64_t>(QuantModeA5::STATIC_QUANT)) 
        && (expandXDesc->GetDataType() != ge::DT_HIFLOAT8) && (expandXDesc->GetDataType() != ge::DT_INT8),
        OP_LOGE(nodeName, "Invalid expandX datatype for quantMode %ld. Only int8/hif8 is supported, but got %s.", 
        quantMode, Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((quantMode == static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) 
        && (expandXDesc->GetDataType() != ge::DT_INT8) && (expandXDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) 
        && (expandXDesc->GetDataType() != ge::DT_FLOAT8_E5M2),
        OP_LOGE(nodeName, "Invalid expandX datatype for quantMode %ld. Only int8/fp8_e4m3fn/fp8_e5m2 is supported, but got %s.", 
        quantMode, Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((quantMode == static_cast<int64_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)) 
        || (quantMode == static_cast<int64_t>(QuantModeA5::MX_QUANT))) && (expandXDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN)
        && (expandXDesc->GetDataType() != ge::DT_FLOAT8_E5M2),
        OP_LOGE(nodeName, "Invalid expandX datatype for quantMode %ld. Only fp8_e4m3fn/fp8_e5m2 is supported, but got %s.", 
        quantMode, Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetContextAttrs(gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchTilingDataA5 &tilingData)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "The attrs is nullptr."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX));

    // 判空
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "The epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "The tpWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "The epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "The tpRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE(nodeName, "The expertShardPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "The sharedExpertRankNumPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "The moeExpertNumPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(nodeName, "The quantModePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "The sharedExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE(nodeName, "The expertTokenNumsType is null."), return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    OP_TILING_CHECK((
        CheckEpAndTpWorldAttrs(nodeName, epWorldSizePtr, tpWorldSizePtr, epRankIdPtr, groupEpPtr) != ge::GRAPH_SUCCESS)
        || (CheckTpRankAttrs(nodeName, tpWorldSizePtr, tpRankIdPtr, groupTpPtr) != ge::GRAPH_SUCCESS), 
        OP_LOGE(nodeName, "Check EP or TP attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((
        CheckExpertAttrs(nodeName, expertShardPtr, sharedExpertRankNumPtr, moeExpertNumPtr, epWorldSizePtr) != ge::GRAPH_SUCCESS)
        || (CheckQuantAndExpertAttrs(nodeName, sharedExpertNumPtr, *quantModePtr, expertTokenNumsTypePtr) != ge::GRAPH_SUCCESS),
        OP_LOGE(nodeName, "Check quant or expert attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDataType(context, nodeName, *quantModePtr) != ge::GRAPH_SUCCESS, 
        OP_LOGE(nodeName, "CheckOutputDataType failed."), return ge::GRAPH_FAILED);
    tilingData.dispatchTilingInfo.set_epWorldSize(static_cast<uint32_t>(*epWorldSizePtr));
    tilingData.dispatchTilingInfo.set_tpWorldSize(static_cast<uint32_t>(*tpWorldSizePtr));
    tilingData.dispatchTilingInfo.set_epRankId(static_cast<uint32_t>(*epRankIdPtr));
    tilingData.dispatchTilingInfo.set_tpRankId(static_cast<uint32_t>(*tpRankIdPtr));
    tilingData.dispatchTilingInfo.set_expertShardType(static_cast<uint32_t>(*expertShardPtr));
    tilingData.dispatchTilingInfo.set_sharedExpertRankNum(static_cast<uint32_t>(*sharedExpertRankNumPtr));
    tilingData.dispatchTilingInfo.set_moeExpertNum(static_cast<uint32_t>(*moeExpertNumPtr));
    tilingData.dispatchTilingInfo.set_quantMode(static_cast<uint32_t>(*quantModePtr));
    tilingData.dispatchTilingInfo.set_expertTokenNumsType(static_cast<uint32_t>(*expertTokenNumsTypePtr));

    return ge::GRAPH_SUCCESS;
}

inline uint32_t CalcRealMode(const gert::TilingContext *context, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    QuantModeA5 quantMode = static_cast<QuantModeA5>(*quantModePtr);
    auto modeToFind = QUANT_MODE_MAP.find({quantMode, static_cast<ge::DataType>(expandXDesc->GetDataType())});
    OP_TILING_CHECK(modeToFind == QUANT_MODE_MAP.end(), 
        OP_LOGE(nodeName, "Failed to find real mode for quantMode=%u and expandX datatype=%s.", 
        static_cast<uint32_t>(quantMode), Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), 
        return static_cast<uint32_t>(RealModeA5::INVALID_MODE));
    OP_LOGD(nodeName, "quantMode=%u, expandX datatype=%s, get realMode=%u\n",
        static_cast<uint32_t>(quantMode), Ops::Base::ToString(expandXDesc->GetDataType()).c_str(), 
        static_cast<uint32_t>(modeToFind->second)); 
    return static_cast<uint32_t>(modeToFind->second);
}

static ge::graphStatus CheckQuantModeAndScales(const gert::TilingContext *context, const char *nodeName,
    bool isScales, const uint32_t quantMode)
{
    OP_TILING_CHECK(isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)),
        OP_LOGE(nodeName, "The scales should be nullptr when quantMode is %u.", 
        quantMode), return ge::GRAPH_FAILED);
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) 
        && ((xDesc->GetDataType() == ge::DT_BF16) || (xDesc->GetDataType() == ge::DT_FLOAT16)),
        OP_LOGE(nodeName, "The scales should be nullptr when quantMode is %u and X datatype is %s.",
        quantMode, Ops::Base::ToString(xDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) 
        && ((xDesc->GetDataType() == ge::DT_HIFLOAT8) || (xDesc->GetDataType() == ge::DT_FLOAT8_E5M2) 
        || (xDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN)),
        OP_LOGE(nodeName, "The scales should not be nullptr when quantMode is %u and X datatype is %s.",
        quantMode, Ops::Base::ToString(xDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)),
        OP_LOGE(nodeName, "The scales should not be nullptr when quantMode is %u.", 
        quantMode), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpWorldSize(const char *nodeName, uint32_t epWorldSize)
{
    // Only the value 4 or 2 is supported currently
    if ( (epWorldSize == DAVID_EP_WORLD_SIZE_FOUR) || (epWorldSize == DAVID_EP_WORLD_SIZE_TWO) ) {
        OP_LOGD(nodeName, "epWorldSize=%u, skip validation\n", epWorldSize);
    } else {
        // 检验epWorldSize是否是8的倍数
        OP_TILING_CHECK(epWorldSize % 8 != 0, OP_LOGE(nodeName,
            "epWorldSize must be a multiple of 8, but got epWorldSize=%u.",
            epWorldSize), return ge::GRAPH_FAILED);
        
        OP_TILING_CHECK((256 % epWorldSize != 0) && (epWorldSize % 144 != 0), OP_LOGE(nodeName,
            "The value of epWorldSize must be in the list[8, 16, 32, 64, 128, 144, 256, 288], but got epWorldSize=%u.",
            epWorldSize), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchTilingDataA5 &tilingData, uint32_t &localMoeExpertNum)
{
    uint32_t epWorldSize = tilingData.dispatchTilingInfo.get_epWorldSize();
    uint32_t tpWorldSize = tilingData.dispatchTilingInfo.get_tpWorldSize();
    uint32_t moeExpertNum = tilingData.dispatchTilingInfo.get_moeExpertNum();
    uint32_t sharedExpertRankNum = tilingData.dispatchTilingInfo.get_sharedExpertRankNum();

    // 校验ep能否均分共享专家
    OP_TILING_CHECK((sharedExpertRankNum != 0) && (epWorldSize % sharedExpertRankNum != 0),
        OP_LOGE(nodeName, "epWorldSize should be non-zero and divisible by sharedExpertRankNum, but epWorldSize=%u, "
        "sharedExpertRankNum=%u.", epWorldSize, sharedExpertRankNum), return ge::GRAPH_FAILED);

    // 校验moe专家数量能否均分给多机
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
        OP_LOGE(nodeName, "The moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
        "but got moeExpertNum=%u, epWorldSize=%u, sharedExpertRankNum=%u.", moeExpertNum, epWorldSize, sharedExpertRankNum),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localMoeExpertNum <= 0, OP_LOGE(nodeName, "The localMoeExpertNum is invalid, localMoeExpertNum=%u",
        localMoeExpertNum), return ge::GRAPH_FAILED);
    // tpWorldSize 当前仅支持1
    OP_TILING_CHECK(tpWorldSize != 1, 
        OP_LOGE(nodeName, "The tpWorldSize must be 1 in current version, but got tpWorldSize=%u.", tpWorldSize), 
        return ge::GRAPH_FAILED);
        
    OP_TILING_CHECK(CheckEpWorldSize(nodeName, epWorldSize) != ge::GRAPH_SUCCESS, 
        OP_LOGE(nodeName, "CheckEpWorldSize failed."), return ge::GRAPH_FAILED);

    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((xDim0 > BS_UPPER_BOUND) || (xDim0 <= 0),
        OP_LOGE(nodeName, "xDim0(BS) is invalid. Should be between [1, %ld], but got xDim0=%ld.", BS_UPPER_BOUND,
                xDim0), return ge::GRAPH_FAILED);
    tilingData.dispatchTilingInfo.set_bs(static_cast<uint32_t>(xDim0));

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBsPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "MoeDistributeDispatch *globalBsPtr=%ld, bs=%ld, epWorldSize=%u\n", *globalBsPtr, xDim0, epWorldSize);
    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < xDim0 * static_cast<int64_t>(epWorldSize)) ||
        ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)), OP_LOGE(nodeName, "globalBS is invalid, only "
        "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, xDim0, epWorldSize), return ge::GRAPH_FAILED);
    if (*globalBsPtr == 0) {
        tilingData.dispatchTilingInfo.set_globalBs(static_cast<uint32_t>(xDim0) * epWorldSize);
    } else {
        tilingData.dispatchTilingInfo.set_globalBs(static_cast<uint32_t>(*globalBsPtr));
    }

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckTwoDimScalesShape(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchTilingDataA5 &tilingData, const int64_t scalesDim0, const int64_t scalesDim1)
{
    uint32_t sharedExpertRankNum = tilingData.dispatchTilingInfo.get_sharedExpertRankNum();   
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.dispatchTilingInfo.get_moeExpertNum());
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    if (sharedExpertRankNum == 0U) {
        OP_TILING_CHECK(scalesDim0 != moeExpertNum, OP_LOGE(nodeName,
            "scales's dim0 not equal to moeExpertNum, scales's dim0=%ld, moeExpertNum=%ld.",
            scalesDim0, moeExpertNum), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(scalesDim0 != (moeExpertNum + 1), OP_LOGE(nodeName,
            "scales's dim0 not equal to moeExpertNum + 1, scales's dim0=%ld, (moeExpertNum + 1)=%ld.",
            scalesDim0, moeExpertNum + 1), return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(xDim1 != scalesDim1, OP_LOGE(nodeName, "scales's dim1 not equal to xShape's dim1, "
        "xShape's dim1=%ld, scales's dim1=%ld.", xDim1, scalesDim1), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckAndSetScalesInfo(gert::TilingContext *context, const char *nodeName, 
    MoeDistributeDispatchTilingDataA5 &tilingData, bool isScales, const uint32_t quantMode)
{
    // 校验scales的维度
    //bs and h have been set in CheckAttrs
    uint32_t h = tilingData.dispatchTilingInfo.get_h();
    uint32_t bs = tilingData.dispatchTilingInfo.get_bs();
    uint64_t scalesRow = 0;
    uint64_t scalesCol = 0;
    uint32_t scalesTypeSize = 0;
    uint64_t scalesCount = 0;
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(nodeName, "scalesShape is null."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return ge::GRAPH_FAILED);
        size_t scalesDimNum = scalesStorageShape->GetStorageShape().GetDimNum();
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        scalesRow = static_cast<uint64_t>(scalesDim0);  
        scalesTypeSize = ge::GetSizeByDataType(scalesDesc->GetDataType());
        if (scalesDimNum == ONE_DIM) {
            // realMode 1 or 9
            OP_TILING_CHECK((quantMode == static_cast<uint32_t>(RealModeA5::STATIC_SCALES)) 
                && (scalesDim0 != h) && (scalesDim0 != STATIC_SCALE_DIM_0),
                OP_LOGE(nodeName, "The expected scalesDim0 is %u or %lu in static quant, but got %ld", 
                h, STATIC_SCALE_DIM_0, scalesDim0), return ge::GRAPH_FAILED);
            OP_TILING_CHECK((quantMode == static_cast<uint32_t>(RealModeA5::HIF8_SCALES)) && (scalesDim0 != HIF8_SCALE_DIM_0),
                OP_LOGE(nodeName, "The expected scalesDim0 is 1 when expandX datatype is hif8 in static quant, but got %ld", 
                scalesDim0), return ge::GRAPH_FAILED);
            scalesCol = ONE_DIM_SCALE_COL_NUM;
            scalesCount = static_cast<uint64_t>(scalesDim0);
        } else if (quantMode == static_cast<uint32_t>(RealModeA5::NO_SCALES)) {
            OP_TILING_CHECK(scalesDim0 != bs,
                OP_LOGE(nodeName, "The expected scalesDim0 is %u when scales is not null in non-quant, but got %ld", 
                bs, scalesDim0), return ge::GRAPH_FAILED);
        } else {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            OP_TILING_CHECK(CheckTwoDimScalesShape(context, nodeName, tilingData, scalesDim0, scalesDim1) != ge::GRAPH_SUCCESS,
                OP_LOGE(nodeName, "CheckTwoDimScalesShape failed."), return ge::GRAPH_FAILED);
            scalesCol = static_cast<uint64_t>(scalesDim1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
        }
    }
    tilingData.dispatchTilingInfo.set_scalesRow(scalesRow);
    tilingData.dispatchTilingInfo.set_scalesCol(scalesCol);
    tilingData.dispatchTilingInfo.set_scalesCount(scalesCount);
    tilingData.dispatchTilingInfo.set_scalesTypeSize(scalesTypeSize);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckExpandXShape(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchTilingDataA5 &tilingData, const int64_t xDim1, uint32_t A)
{
    // 校验expandX的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.dispatchTilingInfo.get_tpWorldSize());
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < tpWorldSize * static_cast<int64_t>(A), 
        OP_LOGE(nodeName, "expandX's dim0 not greater than or equal to A*tpWorldSize, "
        "expandX's dim0=%ld, A*tpWorldSize=%ld.", expandXDim0, tpWorldSize * A), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDim1 != expandXDim1, OP_LOGE(nodeName, "expandX's dim1 not equal to xShape's dim1, "
        "xShape's dim1=%ld, expandX's dim1=%ld.", xDim1, expandXDim1), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckDynamicScalesShape(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchTilingDataA5 &tilingData, const uint32_t quantMode, uint32_t A)
{
    // 校验dynamicScales的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.dispatchTilingInfo.get_tpWorldSize());
    uint64_t h = static_cast<uint64_t>(tilingData.dispatchTilingInfo.get_h());
    if ((quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) 
        && (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        // Dim0
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(dynamicScalesDim0 < static_cast<int64_t>(A) * tpWorldSize, OP_LOGE(nodeName,
            "dynamicScales's dim0 should be equal to or greater than A*tpWorldSize, dynamicScales's dim0=%ld, A*tpWorldSize=%ld.",
            dynamicScalesDim0, A * tpWorldSize), return ge::GRAPH_FAILED);
        // Dim1, only for pergroup and mx
        if (quantMode != static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
            const uint64_t dynamicScalesDim1 = static_cast<uint64_t>(dynamicScalesStorageShape->GetStorageShape().GetDim(1));
            OP_TILING_CHECK((quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) 
                && (dynamicScalesDim1 != ops::CeilDiv(h, MX_BLOCK_SIZE)),
                OP_LOGE(nodeName, "dynamicScales's dim1 should be equal to %lu when quantMode=%u, but got %lu.",
                ops::CeilDiv(h, MX_BLOCK_SIZE), quantMode, dynamicScalesDim1), return ge::GRAPH_FAILED);
            OP_TILING_CHECK((quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) && (dynamicScalesDim1 % 2 != 0), 
                OP_LOGE(nodeName, "dynamicScales's dim1 should be even when quantMode=%u, but got %lu.",
                quantMode, dynamicScalesDim1), return ge::GRAPH_FAILED);
            OP_TILING_CHECK((dynamicScalesDim1 != ops::CeilDiv(h, PERTILE_BLOCK_SIZE)) && 
                (quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)), 
                OP_LOGE(nodeName, "dynamicScales's dim1 should be equal to %lu when quantMode=%u, but got %lu.",
                ops::CeilDiv(h, PERTILE_BLOCK_SIZE), quantMode, dynamicScalesDim1), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckExpandIdxShape(const gert::TilingContext *context, const char *nodeName,
    const int64_t xDim0, const int64_t expertIdsDim1)
{
    // 校验expandIdx的维度
    const gert::StorageShape *expandIdxStorageShape = context->GetOutputShape(OUTPUT_EXPAND_IDX_INDEX);
    const int64_t expandIdxDim0 = expandIdxStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(expandIdxDim0 < expertIdsDim1 * xDim0, OP_LOGE(nodeName,
        "expandIdxDim0 < bs * k, expandIdxDim0=%ld, (bs * k)=%ld.", expandIdxDim0, xDim0 * expertIdsDim1),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckExpertTokenNumsShape(const gert::TilingContext *context, const char *nodeName,
    const bool isSharedExpert, const int64_t localMoeExpertNum)
{
    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    if (isSharedExpert) {
        OP_TILING_CHECK(expertTokenNumsDim0 != 1, OP_LOGE(nodeName, "shared expertTokenNums's dim0 %ld not equal to 1.",
            expertTokenNumsDim0), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(expertTokenNumsDim0 != localMoeExpertNum, OP_LOGE(nodeName,
            "moe expertTokenNums's Dim0 not equal to localMoeExpertNum, expertTokenNumsDim0=%ld, "
            "localMoeExpertNum=%ld.", expertTokenNumsDim0, localMoeExpertNum), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpTpTecvTensorShape(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchTilingDataA5 &tilingData, const bool isSharedExpert, const int64_t localMoeExpertNum)
{
    // 校验epRecvCount和tpRecvCount的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.dispatchTilingInfo.get_tpWorldSize());
    int64_t epWorldSize = static_cast<int64_t>(tilingData.dispatchTilingInfo.get_epWorldSize());
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "epRecvCount is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "tpRecvCount is null."), return ge::GRAPH_FAILED);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpRecvCountDim0 = tpRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = (isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum;
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        epRecvCount *= tpWorldSize;
    }
    OP_TILING_CHECK(epRecvCountDim0 < epRecvCount, OP_LOGE(nodeName,
        "The dimension 0 of epRecvCount should not be less than epWorldSize * localMoeExpertNum * tpWorldSize, "
        "but dimension 0 of epRecvCount=%ld, epWorldSize=%ld, localMoeExpertNum=%ld, tpWorldSize=%ld.",
        epRecvCountDim0, epWorldSize, localMoeExpertNum, tpWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRecvCountDim0 != tpWorldSize, OP_LOGE(nodeName,
        "dimension 0 of tpRecvCount should be equal to tpWorldSize, but dimension 0 of tpRecvCount=%ld, "
        "tpWorldSize=%ld.", tpRecvCountDim0, tpWorldSize), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchTilingDataA5 &tilingData, const bool isSharedExpert, const int64_t localMoeExpertNum)
{
    uint32_t quantMode = tilingData.dispatchTilingInfo.get_quantMode();
    uint32_t A = 0;
    uint32_t globalBs = tilingData.dispatchTilingInfo.get_globalBs();
    uint32_t sharedExpertRankNum = tilingData.dispatchTilingInfo.get_sharedExpertRankNum();
    // 校验输入x的维度1并设h, bs已校验过
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((xDim1 != 7168), OP_LOGE(nodeName, "xShape dims1(H) only supports 7168, but got %ld.", xDim1),
        return ge::GRAPH_FAILED);
    tilingData.dispatchTilingInfo.set_h(static_cast<uint32_t>(xDim1));
    // 校验expert_id的维度并设k
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(nodeName, "expertIdShape is null."), return ge::GRAPH_FAILED);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0, OP_LOGE(nodeName, "xShape's dim0 not equal to expertIdShape's dim0, "
        "xShape's dim0 is %ld, expertIdShape's dim0 is %ld.", xDim0, expertIdsDim0), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX),
        OP_LOGE(nodeName, "expertIdShape's dim1(k) should be in (0, %ld], but got expertIdShape's dim1=%ld.",
        K_MAX, expertIdsDim1), return ge::GRAPH_FAILED);
    tilingData.dispatchTilingInfo.set_k(static_cast<uint32_t>(expertIdsDim1));

    if (isSharedExpert) { // 本卡为共享专家
        A = globalBs / sharedExpertRankNum;
    } else {     // 本卡为moe专家
        A = globalBs * std::min(localMoeExpertNum, expertIdsDim1);
    }
    // 校验expandX、dynamicScales和expandIdx、epSendCount的维度
    OP_TILING_CHECK(CheckExpandXShape(context, nodeName, tilingData, xDim1, A) != ge::GRAPH_SUCCESS 
        || CheckDynamicScalesShape(context, nodeName, tilingData, quantMode, A) != ge::GRAPH_SUCCESS, 
        OP_LOGE(nodeName, "Check expandX or dynamicScales shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckExpandIdxShape(context, nodeName, xDim0, expertIdsDim1) != ge::GRAPH_SUCCESS
        || CheckExpertTokenNumsShape(context, nodeName, isSharedExpert, localMoeExpertNum) != ge::GRAPH_SUCCESS, 
        OP_LOGE(nodeName, "Check expandIdx or expertTokenNums shape failed."), return ge::GRAPH_FAILED);
    // 校验epRecvCount和tpRecvCount的维度
    OP_TILING_CHECK(CheckEpTpTecvTensorShape(context, nodeName, tilingData, isSharedExpert, localMoeExpertNum) != ge::GRAPH_SUCCESS, 
        OP_LOGE(nodeName, "CheckEpTpTecvTensorShape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static void SetHcclTiling(const gert::TilingContext *context, MoeDistributeDispatchTilingDataA5 &tilingData)
{
    tilingData.set_version(HCCL_VERSION);
    tilingData.set_hcommCnt(1);
    const char *nodeName = context->GetNodeName();
    // Temporarily set to int8 to bypass ccu type validation
    tilingData.hcommCfgATA.set_srcDataType(static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(nodeName,
        ge::DT_INT8)));
    tilingData.hcommCfgATA.set_dstDataType(static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(nodeName,
        ge::DT_INT8)));
    tilingData.hcommCfgATA.set_opType(static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_HALFALLTOALLV));
}

static ge::graphStatus GenTilingKey(gert::TilingContext *context, uint32_t realMode, bool isScales)
{
    uint64_t tilingKey = TILING_KEY_BASE_A5;
    uint32_t scalesBit = (isScales ? NUM_10 : NUM_0);
    tilingKey += static_cast<uint64_t>(scalesBit);
    tilingKey += static_cast<uint64_t>(realMode);
    const char *nodeName = context->GetNodeName();
    // Only tpWorldSize 1 is supported currently
    OP_LOGD(nodeName, "tilingKey=%lu", tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static void SetTilingData(gert::TilingContext *context, MoeDistributeDispatchTilingDataA5 &tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

inline ge::graphStatus CheckCommAttrs(const char *nodeName,
    MoeDistributeDispatchTilingDataA5 &tilingData, uint32_t localMoeExpertNum)
{
    uint64_t maxWindowSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    uint64_t h = static_cast<uint64_t>(tilingData.dispatchTilingInfo.get_h());
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData.dispatchTilingInfo.get_epWorldSize());
    uint64_t maxBs = static_cast<uint64_t>(tilingData.dispatchTilingInfo.get_globalBs()) / epWorldSize;
    uint64_t actualSize = epWorldSize * maxBs * h * 2UL * 2UL * static_cast<uint64_t>(localMoeExpertNum);
    if (actualSize > maxWindowSize) {
        OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu, localMoeExpertNum = %u,"
            "ep_worldsize * maxBs * h * 2 * 2 * localMoeExpertNum = %luMB, HCCL_BUFFSIZE=%luMB.", maxBs, h, epWorldSize,
            localMoeExpertNum, actualSize / MB_SIZE + 1UL, maxWindowSize / MB_SIZE);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
inline void SetPlatformInfo(gert::TilingContext* context, MoeDistributeDispatchTilingDataA5 &tilingData, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = 1U;
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    tilingData.dispatchTilingInfo.set_totalUbSize(ubSize);
    tilingData.dispatchTilingInfo.set_aivNum(aivNum);
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
}

ge::graphStatus MoeDistributeDispatchTilingImpl(gert::TilingContext* context)
{
    // Tiling implementation
    OP_TILING_CHECK(context == nullptr, OP_LOGE(OP_NAME, "Fail to get tiling context."), return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE(nodeName, "Fail to get nodeName."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "Start MoeDistributeDispatch tiling.");
    MoeDistributeDispatchTilingDataA5 tilingData;
    uint32_t localMoeExpertNum = 1;
    // Attrs
    OP_TILING_CHECK(GetContextAttrs(context, nodeName, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);
    // Calc real quantMode
    uint32_t quantMode = tilingData.dispatchTilingInfo.get_quantMode();
    uint32_t realMode = CalcRealMode(context, nodeName);
    OP_TILING_CHECK(realMode == static_cast<uint32_t>(RealModeA5::INVALID_MODE), 
        OP_LOGE(nodeName, "CalcRealMode failed."), return ge::GRAPH_FAILED);
    // Scales and Quant
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    tilingData.dispatchTilingInfo.set_isQuant(isScales);
    OP_TILING_CHECK(CheckQuantModeAndScales(context, nodeName, isScales, quantMode) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d, quantMode is %u.",
        static_cast<int32_t>(isScales), quantMode), return ge::GRAPH_FAILED);
    // Output and Input
    OP_TILING_CHECK(
        MoeDistributeDispatchTilingHelper::TilingCheckMoeDistributeDispatchA5(context, nodeName, isScales, quantMode) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);
    // Check Attrs
    OP_TILING_CHECK(CheckAttrs(context, nodeName, tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr failed."), return ge::GRAPH_FAILED);

    bool isSharedExpert = true;
    uint32_t epRankId = tilingData.dispatchTilingInfo.get_epRankId();
    uint32_t sharedExpertRankNum = tilingData.dispatchTilingInfo.get_sharedExpertRankNum();
    if (epRankId >= sharedExpertRankNum) { // 本卡为moe专家
        isSharedExpert = false;
    }
    // Shape
    OP_TILING_CHECK(CheckTensorShape(context, nodeName, tilingData, isSharedExpert, 
        static_cast<int64_t>(localMoeExpertNum)) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetScalesInfo(context, nodeName, tilingData, isScales, realMode) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check scales info failed."), return ge::GRAPH_FAILED);
    // Comm
    OP_TILING_CHECK(CheckCommAttrs(nodeName, tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckCommAttrs failed."), return ge::GRAPH_FAILED);
    tilingData.dispatchTilingInfo.set_totalWinSize(mc2tiling::Mc2TilingUtils::GetMaxWindowSize());
    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    SetHcclTiling(context, tilingData);
    // Tiling Key
    OP_TILING_CHECK(GenTilingKey(context, realMode, isScales) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Fail to get tiling key."), return ge::GRAPH_FAILED);
    // Platform
    SetPlatformInfo(context, tilingData, nodeName);
    PrintTilingDataInfo(nodeName, tilingData);
    // Set Tiling data
    OP_TILING_CHECK(!context->GetRawTilingData(), 
        OP_LOGE(nodeName, "Fail to get raw tiling data. Set tiling data failed."), return ge::GRAPH_FAILED);
    SetTilingData(context, tilingData);
    OP_LOGD(nodeName, "Finish MoeDistributeDispatch tiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTilingA5::DoOpTiling()
{
    return MoeDistributeDispatchTilingImpl(context_);
}

uint64_t MoeDistributeDispatchTilingA5::GetTilingKey() const
{
    // TilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    const char *nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeDispatchTilingA5 get tiling key %lu", tilingKey);
    return tilingKey; 
}

bool MoeDistributeDispatchTilingA5::IsCapable()
{
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND910_95) {
        const char *nodeName = context_->GetNodeName();
        OP_LOGD(nodeName, "Do MoeDistributeDispatchTilingA5 tiling.");
        return true;
    }
    return false;
}
} // namespace optiling
