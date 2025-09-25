/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_tiling_a5.cc
 * \brief
 */

#include "moe_distribute_combine_tiling_arch35.h"


#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <sys/types.h>
#include <queue>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <unistd.h>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "tiling/mc2_tiling_utils.h"
#include "../../../op_kernel/moe_distribute_combine_tiling.h"

namespace {
constexpr uint32_t ATTRS_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTRS_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTRS_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTRS_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTRS_GROUP_TP_INDEX = 4;
constexpr uint32_t ATTRS_TP_WORLD_SIZE_INDEX = 5;
constexpr uint32_t ATTRS_TP_RANK_ID_INDEX = 6;
constexpr uint32_t ATTRS_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTRS_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTRS_SHARED_EXPERT_RANK_NUM_INDEX = 9;
constexpr uint32_t ATTRS_GLOBAL_BS_INDEX = 10;
constexpr uint32_t ATTRS_COMM_QUANT_MODE_INDEX = 12;

const uint64_t TILING_KEY_BASE_A5 = 1000000000000000000;

constexpr uint32_t HCCL_CMD_ALLGATHER = 6U;
constexpr uint32_t HCCL_CMD_ALLTOALLV = 8U;
constexpr uint32_t HCCL_VERSION = 3U;

const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MAX_EP_WORLD_SIZE = 288;
const int64_t MAX_TP_WORLD_SIZE = 2;
const int64_t BS_UPPER_BOUND = 512;

constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
constexpr int64_t K_MAX = 8;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;

constexpr uint32_t DAVID_EP_WORLD_SIZE_FOUR = 4;
constexpr uint32_t DAVID_EP_WORLD_SIZE_TWO = 2;
constexpr uint32_t ENABLE = 1;
constexpr uint32_t NOT_ENABLE = 0;

const std::string OP_NAME = "MoeDistributeCombineA5";
} // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, MoeDistributeCombineTilingDataA5 &tilingData)
{
    OP_LOGD(nodeName, "version %u", tilingData.get_version());
    OP_LOGD(nodeName, "hcommCnt %u", tilingData.get_hcommCnt());

    OP_LOGD(nodeName, "srcDataType %u", tilingData.hcommCfgATA.get_srcDataType());
    OP_LOGD(nodeName, "dstDataType %u", tilingData.hcommCfgATA.get_dstDataType());
    OP_LOGD(nodeName, "opType %u", tilingData.hcommCfgATA.get_opType());

    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.combineTilingInfo.get_epWorldSize());
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.combineTilingInfo.get_tpWorldSize());
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.combineTilingInfo.get_epRankId());
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.combineTilingInfo.get_tpRankId());
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.combineTilingInfo.get_expertShardType());
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.combineTilingInfo.get_sharedExpertRankNum());
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.combineTilingInfo.get_moeExpertNum());
    OP_LOGD(nodeName, "moeExpertPerRankNum is %u.", tilingData.combineTilingInfo.get_moeExpertPerRankNum());
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.combineTilingInfo.get_globalBs());
    OP_LOGD(nodeName, "bs is %d.", tilingData.combineTilingInfo.get_bs());
    OP_LOGD(nodeName, "k is %d.", tilingData.combineTilingInfo.get_k());
    OP_LOGD(nodeName, "h is %d.", tilingData.combineTilingInfo.get_h());
    OP_LOGD(nodeName, "aivNum is %d.", tilingData.combineTilingInfo.get_aivNum());
    OP_LOGD(nodeName, "totalUbSize is %ld.", tilingData.combineTilingInfo.get_totalUbSize());
    OP_LOGD(nodeName, "totalWinSize is %ld.", tilingData.combineTilingInfo.get_totalWinSize());
}

inline ge::graphStatus CheckEpAndTpWorldSize(const char *nodeName, const int64_t *epWorldSizePtr,
                                             const int64_t *tpWorldSizePtr)
{
    OP_TILING_CHECK((*epWorldSizePtr <= 0) || (*epWorldSizePtr > MAX_EP_WORLD_SIZE),
                    OP_LOGE(nodeName, "The valid range of epWorldSize is (0, %ld], but acutually got epWorldSize=%ld.",
                            MAX_EP_WORLD_SIZE, *epWorldSizePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
                    OP_LOGE(nodeName, "The valid range of tpWorldSize is [0, %ld], but acutually got tpWorldSize=%ld.",
                            MAX_TP_WORLD_SIZE, *tpWorldSizePtr),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpRankId(const char *nodeName, const int64_t *epWorldSizePtr, const int64_t *epRankIdPtr)
{
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr),
                    OP_LOGE(nodeName, "The valid range of epRankId is [0, %ld), but acutually got epRankId=%ld.",
                            *epWorldSizePtr, *epRankIdPtr),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckTpRankId(const char *nodeName, const int64_t *tpWorldSizePtr, const int64_t *tpRankIdPtr,
                                     const char *groupTpPtr)
{
    if (*tpWorldSizePtr > 1) {
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
                        OP_LOGE(nodeName, "The valid range of tpRankId is [0, %ld), but acutually got tpRankId=%ld.",
                                *tpWorldSizePtr, *tpRankIdPtr),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr), OP_LOGE(nodeName, "The groupTpPtr is null."), return ge::GRAPH_FAILED);
        uint64_t len = strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH);
        OP_TILING_CHECK(
            (len == 0) || (len == MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "Valid length of groupTp should be in the range (0, %lu), but got strnlen(groupTp)=%lu.",
                    MAX_GROUP_NAME_LENGTH, len),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(*tpRankIdPtr != 0,
                        OP_LOGE(nodeName,
                                "The expected value of tpRankId is 0 in NoTp mode, but the actual value is %ld.",
                                *tpRankIdPtr),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckSharedExpertAttrs(const char *nodeName, const int64_t *sharedExpertRankNumPtr,
                                              const int64_t *epWorldSizePtr, const int64_t *sharedExpertNumPtr)
{
    OP_TILING_CHECK(
        (*sharedExpertRankNumPtr < 0) || (*sharedExpertRankNumPtr >= *epWorldSizePtr),
        OP_LOGE(nodeName,
                "The valid range of sharedExpertRankNum is [0, %ld), but acutually got sharedExpertRankNum=%ld.",
                *epWorldSizePtr, *sharedExpertRankNumPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *sharedExpertNumPtr != 1,
        OP_LOGE(nodeName, "sharedExpertNum only support 1, but got sharedExpertNum=%ld.", *sharedExpertNumPtr),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext *context,
                                               MoeDistributeCombineTilingDataA5 &tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "The context attrs is null."), return ge::GRAPH_FAILED);
    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTRS_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTRS_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTRS_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTRS_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTRS_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTRS_TP_RANK_ID_INDEX);
    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTRS_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTRS_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTRS_MOE_EXPERT_NUM_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTRS_SHARED_EXPERT_NUM_INDEX));
    // 判空
    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE(nodeName, "The groupEpPtr is null."), return ge::GRAPH_FAILED);
    uint64_t len = strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH);
    OP_TILING_CHECK(
        (len == 0) || (len == MAX_GROUP_NAME_LENGTH),
        OP_LOGE(nodeName,
                "Valid length of groupEp must be in the range (0, %lu), but acutually got strnlen(groupEp)=%lu.",
                MAX_GROUP_NAME_LENGTH, len),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "The epWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "The tpWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "The epRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "The tpRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardTypePtr == nullptr, OP_LOGE(nodeName, "The expertShardType is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "The sharedExpertRankNum is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "The moeExpertNum is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "The sharedExpertNum is null."),
                    return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    OP_TILING_CHECK(CheckEpAndTpWorldSize(nodeName, epWorldSizePtr, tpWorldSizePtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckEpAndTpWorldSize failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckEpRankId(nodeName, epWorldSizePtr, epRankIdPtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckEpRankId failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTpRankId(nodeName, tpWorldSizePtr, tpRankIdPtr, groupTpPtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckEpRankId failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*expertShardTypePtr != 0,
                    OP_LOGE(nodeName, "The expected value of expertShardType is 0, but the actual value is %ld.",
                            *expertShardTypePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckSharedExpertAttrs(nodeName, sharedExpertRankNumPtr, epWorldSizePtr, sharedExpertNumPtr) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckEpRankId failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MOE_EXPERT_MAX_NUM),
                    OP_LOGE(nodeName,
                            "The valid range of moeExpertNum is (0, %ld], but acutually got moeExpertNum=%ld.",
                            MOE_EXPERT_MAX_NUM, *moeExpertNumPtr),
                    return ge::GRAPH_FAILED);
    tilingData.combineTilingInfo.set_epWorldSize(static_cast<uint32_t>(*epWorldSizePtr));
    tilingData.combineTilingInfo.set_tpWorldSize(static_cast<uint32_t>(*tpWorldSizePtr));
    tilingData.combineTilingInfo.set_epRankId(static_cast<uint32_t>(*epRankIdPtr));
    tilingData.combineTilingInfo.set_tpRankId(static_cast<uint32_t>(*tpRankIdPtr));
    tilingData.combineTilingInfo.set_expertShardType(static_cast<uint32_t>(*expertShardTypePtr));
    tilingData.combineTilingInfo.set_sharedExpertRankNum(static_cast<uint32_t>(*sharedExpertRankNumPtr));
    tilingData.combineTilingInfo.set_moeExpertNum(static_cast<uint32_t>(*moeExpertNumPtr));
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpWorldSize(const char *nodeName, uint32_t epWorldSize)
{
    // Only the value 4 or 2 is supported currently
    if ((epWorldSize == DAVID_EP_WORLD_SIZE_FOUR) || (epWorldSize == DAVID_EP_WORLD_SIZE_TWO)) {
        OP_LOGD(nodeName, "epWorldSize=%u, skip validation\n", epWorldSize);
    } else {
        // 检验epWorldSize是否是8的倍数
        OP_TILING_CHECK(epWorldSize % 8 != 0,
                        OP_LOGE(nodeName, "epWorldSize should be divisible by 8, but got epWorldSize=%u.", epWorldSize),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (256 % epWorldSize != 0) && (epWorldSize % 144 != 0),
            OP_LOGE(nodeName,
                    "epWorldSize should be in the list[8, 16, 32, 64, 128, 144, 256, 288], but got epWorldSize=%u.",
                    epWorldSize),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static bool CheckAttrs(const gert::TilingContext *context, MoeDistributeCombineTilingDataA5 &tilingData, const char *nodeName,
                       uint32_t &localMoeExpertNum)
{
    uint32_t epWorldSize = tilingData.combineTilingInfo.get_epWorldSize();
    uint32_t tpWorldSize = tilingData.combineTilingInfo.get_tpWorldSize();
    uint32_t moeExpertNum = tilingData.combineTilingInfo.get_moeExpertNum();
    uint32_t sharedExpertRankNum = tilingData.combineTilingInfo.get_sharedExpertRankNum();

    // 校验ep能均分共享
    OP_TILING_CHECK((sharedExpertRankNum != 0) && (epWorldSize % sharedExpertRankNum != 0),
                    OP_LOGE(nodeName,
                            "epWorldSize should be divisible by sharedExpertRankNum, but got epWorldSize=%d, "
                            "sharedExpertRankNum=%d.",
                            epWorldSize, sharedExpertRankNum),
                    return false);
    // 校验moe专家数量能否均分给多机
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
                    OP_LOGE(nodeName,
                            "moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
                            "but got moeExpertNum=%d, epWorldSize=%d, sharedExpertRankNum=%d.",
                            moeExpertNum, epWorldSize, sharedExpertRankNum),
                    return false);
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(localMoeExpertNum <= 0,
                    OP_LOGE(nodeName, "localMoeExpertNum is invalid, localMoeExpertNum = %d", localMoeExpertNum),
                    return false);
    // tpWorldSize 当前仅支持1
    OP_TILING_CHECK(
        tpWorldSize != 1,
        OP_LOGE(nodeName, "The tpWorldSize must be 1 in current version, but got tpWorldSize=%u.", tpWorldSize),
        return false);
    tilingData.combineTilingInfo.set_moeExpertPerRankNum(localMoeExpertNum);
    OP_TILING_CHECK(CheckEpWorldSize(nodeName, epWorldSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckEpWorldSize failed."), return false);

    // 校验输入expertIds的维度0并设bs
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((expertIdsDim0 <= 0) || (expertIdsDim0 > BS_UPPER_BOUND),
                    OP_LOGE(nodeName, "Invalid expertIds dims0(BS) %ld. Should be between [1, %ld].", expertIdsDim0,
                            BS_UPPER_BOUND),
                    return false);
    tilingData.combineTilingInfo.set_bs(static_cast<uint32_t>(expertIdsDim0));

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "The context attrs is null."), return false);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTRS_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBs is null."), return false);
    OP_LOGD(nodeName, "MoeDistributeCombineA5 *globalBsPtr=%ld, bs=%ld, epWorldSize=%u\n", *globalBsPtr, expertIdsDim0,
            epWorldSize);
    OP_TILING_CHECK(
        (*globalBsPtr != 0) && ((*globalBsPtr < static_cast<int64_t>(epWorldSize) * expertIdsDim0) ||
                                ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)),
        OP_LOGE(nodeName,
                "globalBS is invalid, only "
                "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
                "bs=%ld, epWorldSize=%u.",
                *globalBsPtr, expertIdsDim0, epWorldSize),
        return false);
    if (*globalBsPtr == 0) {
        tilingData.combineTilingInfo.set_globalBs(static_cast<uint32_t>(expertIdsDim0) * epWorldSize);
    } else {
        tilingData.combineTilingInfo.set_globalBs(static_cast<uint32_t>(*globalBsPtr));
    }
    return true;
}

inline ge::graphStatus CheckSharedExpertXShape(const gert::TilingContext *context, MoeDistributeCombineTilingDataA5 &tilingData,
    const char *nodeName, int64_t expandXDim1, int64_t expertIdsDim0)
{
    const gert::StorageShape *sharedExpertXShape = context->GetOptionalInputShape(SHARED_EXPERT_X_INDEX);
    uint32_t isSharedExpertX = (sharedExpertXShape != nullptr) ? ENABLE : NOT_ENABLE;
    tilingData.combineTilingInfo.set_hasSharedExpertX(isSharedExpertX);
    if (sharedExpertXShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t sharedExpertXDim0 = sharedExpertXShape->GetStorageShape().GetDim(0);
    int64_t sharedExpertXDim1 = sharedExpertXShape->GetStorageShape().GetDim(1);
    if (sharedExpertXShape->GetStorageShape().GetDimNum() == TWO_DIMS) {
        OP_TILING_CHECK(sharedExpertXDim0 != expertIdsDim0,
            OP_LOGE(nodeName, "sharedExpertX's dim0 not equal to bs, sharedExpertX's dim0 = %ld, bs = %ld",
            sharedExpertXDim0, expertIdsDim0), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(sharedExpertXDim1 != expandXDim1, OP_LOGE(nodeName,
            "sharedExpertX's dim1 not equal to h, sharedExpertX's dim1 = %ld, h = %ld",
            sharedExpertXDim1, expandXDim1), return ge::GRAPH_FAILED);
    } else {
        int64_t sharedExpertXDim2 = sharedExpertXShape->GetStorageShape().GetDim(TWO_DIMS);
        OP_TILING_CHECK(sharedExpertXDim0 * sharedExpertXDim1 != expertIdsDim0,
            OP_LOGE(nodeName, "sharedExpertX's dim0 * sharedExpertX's dim1 not equal to bs, "
            "sharedExpertX's dim0 * sharedExpertX's dim1 = (%ld * %ld), bs = %ld.",
            sharedExpertXDim0, sharedExpertXDim1, expertIdsDim0), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(sharedExpertXDim2 != expandXDim1, OP_LOGE(nodeName,
            "sharedExpertX's dim2 not equal to h, sharedExpertX's dim2 = %ld, h = %ld",
            sharedExpertXDim2, expandXDim1), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckInputTensorShape(const gert::TilingContext *context, MoeDistributeCombineTilingDataA5 &tilingData,
                                             const char *nodeName, bool isShared, int64_t tpWorldSize,
                                             int64_t expertIdsDim0, int64_t expandXDim1,
                                             int64_t expertIdsDim1)
{
    // 校验expandIdx的维度
    const gert::StorageShape *expandIdxStorageShape = context->GetInputShape(EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxStorageShape == nullptr, OP_LOGE(nodeName, "expandIdx is null."), return ge::GRAPH_FAILED);
    int64_t expandIdxDim0 = expandIdxStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(expandIdxDim0 < expertIdsDim0 * expertIdsDim1, OP_LOGE(nodeName,
        "The expandIdxDim0 < bs * k, expandIdxDim0=%ld, (bs * k)=%ld.", expandIdxDim0, expertIdsDim0 * expertIdsDim1),
        return ge::GRAPH_FAILED);

    // 校验epSendCount和tpSendCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.combineTilingInfo.get_epWorldSize());
    int64_t moeExpertPerRankNum = static_cast<int64_t>(tilingData.combineTilingInfo.get_moeExpertPerRankNum());
    const gert::StorageShape *epSendCountStorageShape = context->GetInputShape(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountStorageShape == nullptr, OP_LOGE(nodeName, "epSendCounts is null."),
                    return ge::GRAPH_FAILED);
    const int64_t epSendCountDim0 = epSendCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epSendCount = (isShared) ? epWorldSize : epWorldSize * moeExpertPerRankNum;
    OP_TILING_CHECK(
        epSendCountDim0 < epSendCount * tpWorldSize,
        OP_LOGE(
            nodeName,
            "The epSendCountDim0 not greater than or equal to epSendCount * tpWorldSize, epSendCountDim0=%ld, epSendCount=%ld, \
         tpWorldSize=%ld.",
            epSendCountDim0, epSendCount, tpWorldSize),
        return ge::GRAPH_FAILED);
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        const gert::StorageShape *tpSendCountStorageShape = context->GetOptionalInputShape(TP_SEND_COUNTS_INDEX);
        OP_TILING_CHECK(tpSendCountStorageShape == nullptr, OP_LOGE(nodeName, "tpSendCounts is null."),
                        return ge::GRAPH_FAILED);
        const int64_t tpSendCountDim0 = tpSendCountStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(tpSendCountDim0 != tpWorldSize,
                        OP_LOGE(nodeName,
                                "tpSendCountDim0 not equal to tpWorldSize, tpSendCountDim0=%ld, tpWorldSize=%ld.",
                                tpSendCountDim0, tpWorldSize),
                        return ge::GRAPH_FAILED);
    }

    // 校验expertScales的维度
    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE(nodeName, "expertScales is null."),
                    return ge::GRAPH_FAILED);
    int64_t expertScalesDim0 = expertScalesStorageShape->GetStorageShape().GetDim(0);
    int64_t expertScalesDim1 = expertScalesStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expertScalesDim0 != expertIdsDim0,
                    OP_LOGE(nodeName, "expertScales' dim0 not equal to bs, expertScalesDim0=%ld, bs=%ld",
                            expertScalesDim0, expertIdsDim0),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertScalesDim1 != expertIdsDim1,
                    OP_LOGE(nodeName, "expertScales' dim1 not equal to k, expertScalesDim1=%ld, k=%ld",
                            expertScalesDim1, expertIdsDim1),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckSharedExpertXShape(context, tilingData, nodeName, expandXDim1, expertIdsDim0) !=
        ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "CheckSharedExpertXShape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static bool CheckTensorShape(gert::TilingContext *context, MoeDistributeCombineTilingDataA5 &tilingData,
                             const char *nodeName, bool isShared, uint32_t localExpertNum)
{
    // 校验输入expertIds的维度1并设k, bs已校验过
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);

    uint32_t A = 0;
    uint32_t globalBs = tilingData.combineTilingInfo.get_globalBs();
    uint32_t sharedExpertRankNum = tilingData.combineTilingInfo.get_sharedExpertRankNum();
    if (isShared) { // 本卡为共享专家
        A = globalBs / sharedExpertRankNum;
    } else { // 本卡为moe专家
        A = globalBs * std::min(static_cast<int64_t>(localExpertNum), expertIdsDim1);
    }

    // 校验expandX的维度并设h
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.combineTilingInfo.get_tpWorldSize());
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < tpWorldSize * static_cast<int64_t>(A),
                    OP_LOGE(nodeName,
                            "expandX's dim0 should be greater than or equal to A * tpWorldSize, expandXDim0 = %ld, A = "
                            "%ld, tpWorldSize = %ld",
                            expandXDim0, static_cast<int64_t>(A), tpWorldSize),
                    return false);
    OP_TILING_CHECK((expandXDim1 != 7168),
                    OP_LOGE(nodeName, "expandX dims1(H) only supports 7168, but got %ld.", expandXDim1), return false);
    tilingData.combineTilingInfo.set_h(static_cast<uint32_t>(expandXDim1));

    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX),
                    OP_LOGE(nodeName,
                            "expertIdShape's dim1(k) should be in (0, %ld], but got expertIdShape's dim1=%ld.", K_MAX,
                            expertIdsDim1),
                    return false);
    tilingData.combineTilingInfo.set_k(static_cast<uint32_t>(expertIdsDim1));
    // 校验expandIdx、epSendCount和tpSendCount、expertScales的维度
    OP_TILING_CHECK(CheckInputTensorShape(context, tilingData, nodeName, isShared, tpWorldSize, expertIdsDim0, expandXDim1,
                                          expertIdsDim1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckInputTensorShape failed."), return false);
    // 校验x的维度
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "x is null."), return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0,
                    OP_LOGE(nodeName, "xDim0 not equal to bs, bs=%ld, xDim0=%ld", expertIdsDim0, xDim0), return false);
    OP_TILING_CHECK(xDim1 != expandXDim1,
                    OP_LOGE(nodeName, "xDim1 not equal to h, xDim1=%ld, h=%ld", xDim1, expandXDim1), return false);

    return true;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "get workspace failed"),
                    return ge::GRAPH_FAILED);
    workspace[0] = SYSTEM_NEED_WORKSPACE;
    OP_LOGD(nodeName, "workspce[0] size is %ld", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

static void SetHcclTiling(const gert::TilingContext *context, MoeDistributeCombineTilingDataA5 &tilingData)
{
    tilingData.set_version(HCCL_VERSION);
    tilingData.set_hcommCnt(1);
    const char *nodeName = context->GetNodeName();
    tilingData.hcommCfgATA.set_srcDataType(static_cast<uint32_t>(
        mc2tiling::ConvertGeTypeToHcclType(nodeName, ge::DT_INT8)));
    tilingData.hcommCfgATA.set_dstDataType(static_cast<uint32_t>(
        mc2tiling::ConvertGeTypeToHcclType(nodeName, ge::DT_INT8)));
    tilingData.hcommCfgATA.set_opType(static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_HALFALLTOALLV));
}

static void SetTilingData(gert::TilingContext *context, MoeDistributeCombineTilingDataA5 &tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

inline ge::graphStatus CheckCommAttrs(const char *nodeName,
                                      MoeDistributeCombineTilingDataA5 &tilingData, uint32_t localMoeExpertNum)
{
    uint64_t maxWindowSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    uint64_t h = static_cast<uint64_t>(tilingData.combineTilingInfo.get_h());
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData.combineTilingInfo.get_epWorldSize());
    uint64_t maxBs = static_cast<uint64_t>(tilingData.combineTilingInfo.get_globalBs()) / epWorldSize;
    uint64_t actualSize = epWorldSize * maxBs * h * 2UL * 2UL * static_cast<uint64_t>(localMoeExpertNum);
    if (actualSize > maxWindowSize) {
        OP_LOGE(nodeName,
                "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu, localMoeExpertNum = %u,"
                "ep_worldsize * maxBs * h * 2 * 2 * localMoeExpertNum = %luMB, HCCL_BUFFSIZE=%luMB.",
                maxBs, h, epWorldSize, localMoeExpertNum, actualSize / MB_SIZE + 1UL, maxWindowSize / MB_SIZE);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingImpl(gert::TilingContext *context)
{
    // Tiling implementation
    OP_TILING_CHECK(context == nullptr, OP_LOGE(OP_NAME, "Fail to get tiling context."), return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE(nodeName, "Fail to get nodeName."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "Start MoeDistributeCombineA5 tiling.");
    MoeDistributeCombineTilingDataA5 tilingData;
    bool isShared = true;
    uint32_t localMoeExpertNum = 1;
    // Attrs
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, tilingData, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);
    // Output and Input
    OP_TILING_CHECK(MoeDistributeCombineTilingHelper::TilingCheckMoeDistributeCombine(context, nodeName) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check params failed"), return ge::GRAPH_FAILED);
    // Check Attrs
    OP_TILING_CHECK(!CheckAttrs(context, tilingData, nodeName, localMoeExpertNum),
                    OP_LOGE(nodeName, "attr check failed."), return ge::GRAPH_FAILED);

    uint32_t sharedExpertRankNum = tilingData.combineTilingInfo.get_sharedExpertRankNum();
    uint32_t epRankId = tilingData.combineTilingInfo.get_epRankId();
    if (epRankId >= sharedExpertRankNum) { // 本卡为moe专家
        isShared = false;
    }
    // Shape
    OP_TILING_CHECK(!CheckTensorShape(context, tilingData, nodeName, isShared, localMoeExpertNum),
                    OP_LOGE(nodeName, "param dim check failed."), return ge::GRAPH_FAILED);
    // Comm
    OP_TILING_CHECK(CheckCommAttrs(nodeName, tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckCommAttrs failed."), return ge::GRAPH_FAILED);
    tilingData.combineTilingInfo.set_totalWinSize(mc2tiling::Mc2TilingUtils::GetMaxWindowSize());
    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(nodeName, "Tiling set workspace Failed"), return ge::GRAPH_FAILED);
    SetHcclTiling(context, tilingData);
    // Tiling Key support only 1 scenario in current version
    uint64_t tilingKey = TILING_KEY_BASE_A5;
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    // Platform
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    tilingData.combineTilingInfo.set_aivNum(aivNum);
    tilingData.combineTilingInfo.set_totalUbSize(ubSize);
    OP_LOGD(nodeName, "blockdim = %u, aivNum = %lu, ubsize = %lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, tilingData);
    // Set Tiling data
    OP_TILING_CHECK(!context->GetRawTilingData(),
                    OP_LOGE(nodeName, "Fail to get raw tiling data. Set tiling data failed."), return ge::GRAPH_FAILED);
    SetTilingData(context, tilingData);
    OP_LOGD(nodeName, "Finish MoeDistributeCombine tiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingA5::DoOpTiling()
{
    return MoeDistributeCombineTilingImpl(context_);
}

uint64_t MoeDistributeCombineTilingA5::GetTilingKey() const
{
    // TilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    const char *nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombineTilingA5 get tiling key %lu", tilingKey);
    return tilingKey;
}

bool MoeDistributeCombineTilingA5::IsCapable()
{
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND910_95) {
        const char *nodeName = context_->GetNodeName();
        OP_LOGD(nodeName, "Do MoeDistributeCombineTilingA5 tiling.");
        return true;
    }
    return false;
}

} // namespace optiling
