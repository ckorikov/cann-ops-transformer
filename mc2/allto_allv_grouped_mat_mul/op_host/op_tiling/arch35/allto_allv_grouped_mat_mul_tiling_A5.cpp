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
 * \file allto_allv_grouped_mat_mul_tiling_A5.cc
 * \brief
 */

#include "allto_allv_grouped_mat_mul_tiling_A5.h"
#include <string>
#include <numeric>
#include <climits>
#include "mc2_hcom_topo_info.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "tiling/hccl_formulaic_tiling.h"
#include "tiling/mc2_tiling_utils.h"
#include "tiling/mc2_tiling_common_var.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "../allto_allv_grouped_mat_mul_tiling_base.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

constexpr uint32_t GMM_X_INDEX = 0U;
constexpr uint32_t GMM_WEIGHT_INDEX = 1U;
constexpr uint32_t SEND_COUNTS_TENSOR_INDEX = 2U;
constexpr uint32_t RECV_COUNTS_TENSOR_INDEX = 3U;
constexpr uint32_t MM_X_INDEX = 4U;
constexpr uint32_t MM_WEIGHT_INDEX = 5U;

constexpr uint32_t OUTPUT_GMM_Y_INDEX = 0U;
constexpr uint32_t OUTPUT_MM_Y_INDEX = 1U;
constexpr uint32_t OUTPUT_PERMUTE_OUT_INDEX = 2U;

constexpr uint64_t DIM_TWO = 2UL;
constexpr uint64_t DIM_ONE = 1UL;
constexpr uint64_t DIM_THREE = 3UL;

constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_EIGHT = 8;
constexpr int64_t NUM_SIXTEEN = 16;
constexpr int64_t NUM_THIRTYTWO = 32;
constexpr int64_t NUM_SIXTYFOUR = 64;
constexpr int64_t MAX_EXPERT_NUM = 256;
constexpr int64_t MAX_BSK = 52428800;
constexpr int64_t MAX_SHAPE_SIZE = 65536;
constexpr int64_t MAX_SHARED_H_SHAPE_SIZE = 12288;
constexpr int64_t MAX_DIM_VALUE = 65536;
constexpr int64_t MAX_EXPERT_NUM_PER_RANK = 32;

constexpr uint32_t HCCL_CMD_ALLGATHER = 6U;
constexpr uint32_t HCCL_CMD_ALLTOALLV = 8U;
constexpr uint32_t HCCL_VERSION = 3U;

constexpr uint32_t ATTR_GROUP_INDEX = 0U;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1U;
constexpr uint32_t ATTR_SEND_COUNTS_INDEX = 2U;
constexpr uint32_t ATTR_RECV_COUNTS_INDEX = 3U;
constexpr uint32_t ATTR_TRANS_GMM_WEIGHT_INDEX = 4U;
constexpr uint32_t ATTR_TRANS_MM_WEIGHT_INDEX = 5U;
constexpr uint32_t ATTR_PERMUTE_OUT_FLAG_INDEX = 6U;

constexpr uint64_t TILINGKEY_FP16 = 1000UL;
constexpr uint64_t TILINGKEY_BF16 = 0UL;
constexpr uint64_t TILINGKEY_MM = 100UL;
constexpr uint64_t TILINGKEY_GMM_WEIGHT_TRANSPOSE = 10UL;
constexpr uint64_t TILINGKEY_MM_WEIGHT_TRANSPOSE = 1UL;
constexpr uint64_t TILINGKEY_A5_OFFSET = 1000000000000000000UL;

constexpr int64_t MAX_E = 128L;
constexpr int64_t MAX_MATMUL_K = 65535L;

constexpr int64_t BEST_L1_PARTA = 256L * 1024L;
constexpr int64_t BEST_L1_PARTB = 128L * 1024L;
constexpr int64_t BEST_BASE_N = 256L;
constexpr uint32_t UB_DEVIDE_NUM = 2U;
constexpr uint32_t UB_CALSIZE_PER_BLOCK = 16U * 1024U;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2UL;
constexpr uint64_t DOUBEL_BUFFER_STEPKA_STEPKB = 2UL;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;
constexpr uint32_t MAX_TURN_NUM = 24U;
constexpr uint32_t MAX_BASE_K = 128U;
constexpr uint64_t COMM_TILE = 8; // 每卡数据分配几次计算

const char* A5_INNER_DEBUG = "AlltoAllvGroupedMatMul Tiling";

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15U; // 15: 16 bytes up-align
    }
    return a & ~15U; // ~15: 16bytes down-align
}

static inline uint32_t Ceil(uint32_t a, uint32_t b)
{
    if (b == 0U) {
        return a;
    }
    return (a + b - 1U) / b;
}

static uint64_t GMMGetSizePlatForm(
    const platform_ascendc::CoreMemType memType, platform_ascendc::PlatformAscendC ascendcPlatform)
{
    uint64_t size = 0UL;
    ascendcPlatform.GetCoreMemSize(memType, size);
    return size;
}

static void PrintTilingData(AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    OP_LOGD(A5_INNER_DEBUG, "version %u", tilingData.get_version());
    OP_LOGD(A5_INNER_DEBUG, "hcommCnt %u", tilingData.get_hcommCnt());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo BSK %lu", tilingData.commonTilingInfo.get_BSK());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo BS %lu", tilingData.commonTilingInfo.get_BS());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo K %lu", tilingData.commonTilingInfo.get_K());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo H1 %lu", tilingData.commonTilingInfo.get_H1());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo H2 %lu", tilingData.commonTilingInfo.get_H2());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo A %lu", tilingData.commonTilingInfo.get_A());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo N1 %lu", tilingData.commonTilingInfo.get_N1());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo N2 %lu", tilingData.commonTilingInfo.get_N2());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo epWorldSize %lu", tilingData.commonTilingInfo.get_epWorldSize());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo stepSize %lu", tilingData.commonTilingInfo.get_stepSize());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo E_ep %lu", tilingData.commonTilingInfo.get_E_ep());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo commOut %lu", tilingData.commonTilingInfo.get_commOut());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo aivCoreNum %lu", tilingData.commonTilingInfo.get_aivCoreNum());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo aicCoreNum %lu", tilingData.commonTilingInfo.get_aicCoreNum());
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo totalUbSize %lu", tilingData.commonTilingInfo.get_totalUbSize());
    OP_LOGD(
        A5_INNER_DEBUG, "commonTilingInfo isGmmWeightTrans %u",
        tilingData.commonTilingInfo.get_isGmmWeightTrans() ? 1U : 0U);
    OP_LOGD(
        A5_INNER_DEBUG, "commonTilingInfo isMmWeightTrans%u",
        tilingData.commonTilingInfo.get_isMmWeightTrans() ? 1U : 0U);
    OP_LOGD(
        A5_INNER_DEBUG, "commonTilingInfo isSendCntsTensor %u",
        tilingData.commonTilingInfo.get_isSendCntsTensor() ? 1U : 0U);
    OP_LOGD(
        A5_INNER_DEBUG, "commonTilingInfo isRecvCntsTensor %u",
        tilingData.commonTilingInfo.get_isRecvCntsTensor() ? 1U : 0U);
    OP_LOGD(
        A5_INNER_DEBUG, "commonTilingInfo isPermuteOut %u", tilingData.commonTilingInfo.get_isPermuteOut() ? 1U : 0U);
    OP_LOGD(A5_INNER_DEBUG, "commonTilingInfo isNeedMM%u", tilingData.commonTilingInfo.get_isNeedMM() ? 1U : 0U);
    Mc2Log::PrintTCubeTilingData(A5_INNER_DEBUG, tilingData.gmmTilingData);
    if (tilingData.commonTilingInfo.get_isNeedMM()) {
        OP_LOGD(A5_INNER_DEBUG, "Has shared expert.");
        Mc2Log::PrintTCubeTilingData(A5_INNER_DEBUG, tilingData.mmTilingData);
    }
}

struct PlatFormMemSize {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;

    explicit PlatFormMemSize(platform_ascendc::PlatformAscendC ascendcPlatform)
        : ubSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::UB, ascendcPlatform)),
          l1Size(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L1, ascendcPlatform)),
          l0CSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_C, ascendcPlatform)),
          l0ASize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_A, ascendcPlatform)),
          l0BSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_B, ascendcPlatform))
    {}
};

static bool CheckCntSum(
    const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    auto gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    auto attrs = context->GetAttrs();
    auto y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);

    int64_t a = y->GetStorageShape().GetDim(0);
    int64_t bsK = gmmX->GetStorageShape().GetDim(0);
    int64_t eOverEp = gmmWeight->GetStorageShape().GetDim(0);
    int64_t epWorldSize = static_cast<int64_t>(tilingData.commonTilingInfo.get_epWorldSize());

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    const int64_t* sendArray = static_cast<const int64_t*>(sendCountsPtr->GetData());
    const int64_t* recvArray = static_cast<const int64_t*>(recvCountsPtr->GetData());
    
    for (int64_t i = 1; i <= epWorldSize; i++) {
        for (int64_t j = (i - 1) * eOverEp; j <= i * eOverEp - 1; j++) {
            OP_TILING_CHECK(
                (sendArray[j] < NUM_ZERO) || (sendArray[j] > bsK),
                OP_LOGE(A5_INNER_DEBUG, "sendCounts[%ld] should be in [0, bsK[%ld]], but get %ld",j, bsK, sendArray[j]),
                return false);
            OP_TILING_CHECK(
                (recvArray[j] < NUM_ZERO) || (recvArray[j] > a),
                OP_LOGE(A5_INNER_DEBUG, "recvCounts[%ld] should be in [0, a[%ld]], but get %ld",j, a, recvArray[j]),
                return false);
        }
        }
        return true;
}

static bool CheckSendCntAndRecvCnt(
    const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    auto attrs = context->GetAttrs();
    auto gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    auto y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    
    int64_t eOverEp = gmmWeight->GetStorageShape().GetDim(0);
    int64_t bsK = gmmX->GetStorageShape().GetDim(0);

    int64_t epWorldSize = static_cast<int64_t>(tilingData.commonTilingInfo.get_epWorldSize());
    int64_t a = y->GetStorageShape().GetDim(0);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    size_t recvSize = recvCountsPtr->GetSize();
    const int64_t* recvArray = static_cast<const int64_t*>(recvCountsPtr->GetData());
    size_t sendSize = sendCountsPtr->GetSize();
    const int64_t* sendArray = static_cast<const int64_t*>(sendCountsPtr->GetData());
    OP_TILING_CHECK(
        static_cast<int64_t>(recvSize) != epWorldSize * eOverEp,
        OP_LOGE(
            A5_INNER_DEBUG, "The length of recvCnts[%lu] should be equal to eOverEp * epworldSize[%ld]", recvSize,
            epWorldSize * eOverEp),
        return false);
    OP_TILING_CHECK(
        static_cast<int64_t>(sendSize) != epWorldSize * eOverEp,
        OP_LOGE(
            A5_INNER_DEBUG, "The length of sendCnts[%lu] should be equal to eOverEp * epworldSize[%ld]", sendSize,
            epWorldSize * eOverEp),
        return false);

    int64_t recvSum = 0;
    for (uint64_t i = 0; i < recvSize; i++) {
        recvSum += recvArray[i];
    }
    OP_TILING_CHECK(
        a != recvSum,
        OP_LOGE(A5_INNER_DEBUG, "a[%ld] should be equal to the sum of recvCounts[%ld]!", a, recvSum),
        return false);

    int64_t sendSum = 0;
    for (uint64_t i = 0; i < sendSize; i++) {
        sendSum += sendArray[i];
    }
    OP_TILING_CHECK(
        bsK != sendSum, OP_LOGE(A5_INNER_DEBUG, "bsK[%ld] should be equal to the sum of sendCounts[%ld]!", bsK, sendSum),
        return false);
    OP_TILING_CHECK(
        !CheckCntSum(context, tilingData),
        OP_LOGE(A5_INNER_DEBUG, "CheckCntSum failed!"), return false);
    return true;
}

static bool CheckDType(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    OP_TILING_CHECK(
        (context->GetInputDesc(GMM_X_INDEX) == nullptr), 
        OP_LOGE(
            A5_INNER_DEBUG, "gmm_x is nullptr."), 
        return false);
    OP_TILING_CHECK(
        (context->GetInputDesc(GMM_WEIGHT_INDEX) == nullptr), 
        OP_LOGE(
            A5_INNER_DEBUG, "gmm_weight is nullptr."), 
        return false);
    OP_TILING_CHECK(
        (context->GetOutputDesc(OUTPUT_GMM_Y_INDEX) == nullptr), 
        OP_LOGE(
            A5_INNER_DEBUG, "gmm_y is nullptr."), 
        return false);

    auto gmmXDType = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    auto gmmWeightDType = context->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    auto yDType = context->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();

    OP_TILING_CHECK(
        !((gmmXDType == ge::DT_FLOAT16) || (gmmXDType == ge::DT_BF16)),
        OP_LOGE(
            A5_INNER_DEBUG, "GmmX DataType [%s] is not supported.",
            TypeUtils::DataTypeToSerialString(gmmXDType).c_str()),
        return false);
    OP_TILING_CHECK(
        !((gmmWeightDType == ge::DT_FLOAT16) || (gmmWeightDType == ge::DT_BF16)),
        OP_LOGE(
            A5_INNER_DEBUG, "GmmWeight DataType [%s] is not supported.",
            TypeUtils::DataTypeToSerialString(gmmWeightDType).c_str()),
        return false);
    OP_TILING_CHECK(
        !((yDType == ge::DT_FLOAT16) || (yDType == ge::DT_BF16)),
        OP_LOGE(
            A5_INNER_DEBUG, "GmmY DataType [%s] is not supported.", TypeUtils::DataTypeToSerialString(yDType).c_str()),
        return false);
    OP_TILING_CHECK(
        !((gmmXDType == gmmWeightDType) && (gmmXDType == yDType)),
        OP_LOGE(
            A5_INNER_DEBUG, "GmmX [%s], GmmWeight [%s], GmmY [%s] should be same.",
            TypeUtils::DataTypeToSerialString(gmmXDType).c_str(),
            TypeUtils::DataTypeToSerialString(gmmWeightDType).c_str(),
            TypeUtils::DataTypeToSerialString(yDType).c_str()),
        return false);

    if (tilingData.commonTilingInfo.get_isSendCntsTensor()) {
        auto sendCntsDType = context->GetOptionalInputDesc(SEND_COUNTS_TENSOR_INDEX)->GetDataType();
        OP_TILING_CHECK(
            !((sendCntsDType == ge::DT_INT32) || (sendCntsDType == ge::DT_INT64)),
            OP_LOGE(
                A5_INNER_DEBUG, "SendCountsTensor DataType [%s] is not supported.",
                TypeUtils::DataTypeToSerialString(sendCntsDType).c_str()),
            return false);
    }
    if (tilingData.commonTilingInfo.get_isRecvCntsTensor()) {
        auto recvCntsDType = context->GetOptionalInputDesc(SEND_COUNTS_TENSOR_INDEX)->GetDataType();
        OP_TILING_CHECK(
            !((recvCntsDType == ge::DT_INT32) || (recvCntsDType == ge::DT_INT64)),
            OP_LOGE(
                A5_INNER_DEBUG, "RecvCountsTensor DataType [%s] is not supported.",
                TypeUtils::DataTypeToSerialString(recvCntsDType).c_str()),
            return false);
    }

    if (tilingData.commonTilingInfo.get_isNeedMM()) {
        OP_TILING_CHECK(
            (context->GetOptionalInputDesc(MM_X_INDEX) == nullptr), 
            OP_LOGE(
                A5_INNER_DEBUG, "mm_x is nullptr."), 
            return false);
        OP_TILING_CHECK(
            (context->GetOptionalInputDesc(MM_WEIGHT_INDEX) == nullptr), 
            OP_LOGE(
                A5_INNER_DEBUG, "mm_weight is nullptr."), 
            return false);
        OP_TILING_CHECK(
            (context->GetOutputDesc(OUTPUT_MM_Y_INDEX) == nullptr), 
            OP_LOGE(
                A5_INNER_DEBUG, "mm_y is nullptr."), 
            return false);
        auto mmXDType = context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType();
        auto mmWeightDType = context->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType();
        auto mmYDType = context->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType();
        OP_TILING_CHECK(
            !((gmmXDType == mmXDType) && (gmmXDType == mmWeightDType) && (gmmXDType == mmYDType)),
            OP_LOGE(
                A5_INNER_DEBUG, "MmX [%s], MmWeight [%s], MmY [%s] should be same as GMM [%s].",
                TypeUtils::DataTypeToSerialString(mmXDType).c_str(),
                TypeUtils::DataTypeToSerialString(mmWeightDType).c_str(),
                TypeUtils::DataTypeToSerialString(mmYDType).c_str(),
                TypeUtils::DataTypeToSerialString(gmmXDType).c_str()),
            return false);
    }

    if (tilingData.commonTilingInfo.get_isPermuteOut()) {
        OP_TILING_CHECK(
            (context->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX) == nullptr), 
            OP_LOGE(
                A5_INNER_DEBUG, "permute_out is nullptr."), 
            return false);
        auto permuteOutDtype = context->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX)->GetDataType();
        OP_TILING_CHECK(
            !(gmmXDType == permuteOutDtype),
            OP_LOGE(
                A5_INNER_DEBUG, "PermuteOut Dataype [%s] should be same as GMM [%s].",
                TypeUtils::DataTypeToSerialString(permuteOutDtype).c_str(),
                TypeUtils::DataTypeToSerialString(gmmXDType).c_str()),
            return false);
    }

    return true;
}

static bool CheckFormat(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    auto gmmXFormat = context->GetInputDesc(GMM_X_INDEX)->GetFormat().GetStorageFormat();
    auto gmmWeightFormat = context->GetInputDesc(GMM_WEIGHT_INDEX)->GetFormat().GetStorageFormat();
    auto yFormat = context->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetFormat().GetStorageFormat();

    OP_TILING_CHECK(!(gmmXFormat == ge::FORMAT_ND), OP_LOGE(A5_INNER_DEBUG, "GmmX Format should be ND."), return false);
    OP_TILING_CHECK(
        !(gmmWeightFormat == ge::FORMAT_ND), OP_LOGE(A5_INNER_DEBUG, "GmmWeight Format should be ND."), return false);
    OP_TILING_CHECK(!(yFormat == ge::FORMAT_ND), OP_LOGE(A5_INNER_DEBUG, "GmmY Format should be ND."), return false);

    if (tilingData.commonTilingInfo.get_isSendCntsTensor()) {
        auto sendCntsFormat = context->GetOptionalInputDesc(SEND_COUNTS_TENSOR_INDEX)->GetFormat().GetStorageFormat();
        OP_TILING_CHECK(
            !(sendCntsFormat == ge::FORMAT_ND), OP_LOGE(A5_INNER_DEBUG, "SendCountsTensor Format should be ND."),
            return false);
    }

    if (tilingData.commonTilingInfo.get_isRecvCntsTensor()) {
        auto recvCntsFormat = context->GetOptionalInputDesc(SEND_COUNTS_TENSOR_INDEX)->GetFormat().GetStorageFormat();
        OP_TILING_CHECK(
            !(recvCntsFormat == ge::FORMAT_ND), OP_LOGE(A5_INNER_DEBUG, "RecvCountsTensor Format should be ND."),
            return false);
    }

    if (tilingData.commonTilingInfo.get_isNeedMM()) {
        auto mmXFormat = context->GetOptionalInputDesc(MM_X_INDEX)->GetFormat().GetStorageFormat();
        auto mmWeightFormat = context->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetFormat().GetStorageFormat();
        auto mmYFormat = context->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetFormat().GetStorageFormat();
        OP_TILING_CHECK(
            !(mmXFormat == ge::FORMAT_ND), OP_LOGE(A5_INNER_DEBUG, "MmX Format should be ND."), return false);
        OP_TILING_CHECK(
            !(mmWeightFormat == ge::FORMAT_ND), OP_LOGE(A5_INNER_DEBUG, "MmWeight Format should be ND."), return false);
        OP_TILING_CHECK(
            !(mmYFormat == ge::FORMAT_ND), OP_LOGE(A5_INNER_DEBUG, "MmY Format should be ND."), return false);
    }
    return true;
}

static bool CheckDimNum(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    auto gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    auto y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);

    OP_TILING_CHECK(
        gmmX->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE(A5_INNER_DEBUG, "GmmX's Dim should be 2, but get %lu.", gmmX->GetStorageShape().GetDimNum()),
        return false);
    OP_TILING_CHECK(
        gmmWeight->GetStorageShape().GetDimNum() != DIM_THREE,
        OP_LOGE(A5_INNER_DEBUG, "GmmWeight's Dim should be 3, but get %lu.", gmmWeight->GetStorageShape().GetDimNum()),
        return false);
    OP_TILING_CHECK(
        y->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE(A5_INNER_DEBUG, "Y's Dim should be 2, but get %lu.", y->GetStorageShape().GetDimNum()), return false);

    if (tilingData.commonTilingInfo.get_isSendCntsTensor()) {
        auto sendCounts = context->GetOptionalInputShape(SEND_COUNTS_TENSOR_INDEX);
        OP_TILING_CHECK(
            sendCounts->GetStorageShape().GetDimNum() != DIM_ONE,
            OP_LOGE(
                A5_INNER_DEBUG, "SendCountsTensor's Dim should be 1, but get %lu.",
                sendCounts->GetStorageShape().GetDimNum()),
            return false);
    }
    if (tilingData.commonTilingInfo.get_isRecvCntsTensor()) {
        auto recvCounts = context->GetOptionalInputShape(RECV_COUNTS_TENSOR_INDEX);
        OP_TILING_CHECK(
            recvCounts->GetStorageShape().GetDimNum() != DIM_ONE,
            OP_LOGE(
                A5_INNER_DEBUG, "RecvCountsTensor's Dim should be 1, but get %lu.",
                recvCounts->GetStorageShape().GetDimNum()),
            return false);
    }

    if (tilingData.commonTilingInfo.get_isNeedMM()) {
        auto mmX = context->GetOptionalInputShape(MM_X_INDEX);
        auto mmWeight = context->GetOptionalInputShape(MM_WEIGHT_INDEX);
        auto mmY = context->GetOutputShape(OUTPUT_MM_Y_INDEX);
        OP_TILING_CHECK(
            mmX->GetStorageShape().GetDimNum() != DIM_TWO,
            OP_LOGE(A5_INNER_DEBUG, "MmX's Dim should be 2, but get %lu.", mmX->GetStorageShape().GetDimNum()),
            return false);
        OP_TILING_CHECK(
            mmWeight->GetStorageShape().GetDimNum() != DIM_TWO,
            OP_LOGE(
                A5_INNER_DEBUG, "MmWeight's Dim should be 2, but get %lu.", mmWeight->GetStorageShape().GetDimNum()),
            return false);
        OP_TILING_CHECK(
            mmY->GetStorageShape().GetDimNum() != DIM_TWO,
            OP_LOGE(A5_INNER_DEBUG, "MmY's Dim should be 2, but get %lu.", mmY->GetStorageShape().GetDimNum()),
            return false);
    }

    return true;
}

static bool CheckSharedExpDimValue(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    if (!tilingData.commonTilingInfo.get_isNeedMM()) {
        return true;
    }
    const gert::StorageShape* mmX = context->GetOptionalInputShape(MM_X_INDEX);
    const gert::StorageShape* mmWeight = context->GetOptionalInputShape(MM_WEIGHT_INDEX);
    const gert::StorageShape* mmY = context->GetOutputShape(OUTPUT_MM_Y_INDEX);

    int64_t bs = mmX->GetStorageShape().GetDim(0);
    int64_t h2 = mmX->GetStorageShape().GetDim(1);
    int64_t mmWeightH = tilingData.commonTilingInfo.get_isMmWeightTrans() ? mmWeight->GetStorageShape().GetDim(1) :
                                                                            mmWeight->GetStorageShape().GetDim(0);
    int64_t n2 = tilingData.commonTilingInfo.get_isMmWeightTrans() ? mmWeight->GetStorageShape().GetDim(0) :
                                                                     mmWeight->GetStorageShape().GetDim(1);
    int64_t mmYDim0 = mmY->GetStorageShape().GetDim(0);
    int64_t mmYDim1 = mmY->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(
        h2 != mmWeightH, OP_LOGE(A5_INNER_DEBUG, "mmWeightDim0[%ld] should equal to h2[%ld].", mmWeightH, h2),
        return false);
    OP_TILING_CHECK(
        n2 != mmYDim1, OP_LOGE(A5_INNER_DEBUG, "mmYDim1[%ld] should equal to n2[%ld].", mmYDim1, n2), return false);
    OP_TILING_CHECK(
        mmYDim0 != bs, OP_LOGE(A5_INNER_DEBUG, "mmYDim0[%ld] should equal to bs[%ld].", mmYDim0, bs), return false);
    OP_TILING_CHECK(
        h2 > MAX_MATMUL_K, OP_LOGE(A5_INNER_DEBUG, "h[%ld] exceeds matmul's limit[%ld]", h2, MAX_MATMUL_K),
        return false);
    return true;
}

static bool CheckDimValueShape(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    auto gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    
    int64_t bsK = gmmX->GetStorageShape().GetDim(0);
    int64_t h = gmmX->GetStorageShape().GetDim(1);
    int64_t eOverEp = gmmWeight->GetStorageShape().GetDim(0);
    int64_t gmmWeightH = tilingData.commonTilingInfo.get_isGmmWeightTrans() ? gmmWeight->GetStorageShape().GetDim(2) :
                                                                              gmmWeight->GetStorageShape().GetDim(1);
    int64_t n1 = tilingData.commonTilingInfo.get_isGmmWeightTrans() ? gmmWeight->GetStorageShape().GetDim(1) :
                                                                      gmmWeight->GetStorageShape().GetDim(2);
    int64_t epWorldSize = static_cast<int64_t>(tilingData.commonTilingInfo.get_epWorldSize());
    OP_TILING_CHECK(
        ((bsK <= NUM_ZERO) || (bsK >= MAX_BSK)), OP_LOGE(A5_INNER_DEBUG, "bsK[%ld] should be in (0, 52428800)!", bsK),
        return false);

    OP_TILING_CHECK(
        ((h <= NUM_ZERO) || (h >= MAX_DIM_VALUE)), OP_LOGE(A5_INNER_DEBUG, "h1[%ld] should be in (0, 65536)!", h),
        return false);

    OP_TILING_CHECK(
        ((gmmWeightH <= NUM_ZERO) || (gmmWeightH > MAX_SHARED_H_SHAPE_SIZE)),
        OP_LOGE(A5_INNER_DEBUG, "h2[%ld] should be in (0, 12288]!", h),
        return false);

    OP_TILING_CHECK(
        ((n1 <= NUM_ZERO) || (n1 >= MAX_DIM_VALUE)), OP_LOGE(A5_INNER_DEBUG, "n1[%ld] should be in (0, 65536)!", n1),
        return false);
        
    OP_TILING_CHECK(
        ((eOverEp <= NUM_ZERO) || (eOverEp > MAX_EXPERT_NUM_PER_RANK)),
        OP_LOGE(A5_INNER_DEBUG, "eOverEp[%ld] should be in (0, 32]!", eOverEp), return false);

    OP_TILING_CHECK(
        !CheckSendCntAndRecvCnt(context, tilingData),
        OP_LOGE(A5_INNER_DEBUG, "CheckSendCntAndRecvCnt failed!"), return false);
    
    std::vector<int64_t> epWorldSizeOptional{2, 4, 8, 16, 32, 64};
    OP_TILING_CHECK(
        std::find(epWorldSizeOptional.begin(), epWorldSizeOptional.end(), epWorldSize) == epWorldSizeOptional.end(),
        OP_LOGE(A5_INNER_DEBUG, "epWorldSize[%ld] should be 2,4,8,16,32,64!", epWorldSize), return false);
    return true;
}

static bool CheckDimValueIsNeedMM(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    int64_t bsK = gmmX->GetStorageShape().GetDim(0);

    if (tilingData.commonTilingInfo.get_isNeedMM()) {
        int64_t bs = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(0);
        if ((bs <= NUM_ZERO) || (bs >= MAX_SHAPE_SIZE)) {
            OP_LOGE(A5_INNER_DEBUG, "bs should be in (0, 52428800), but got %lu!", bs);
            return false;
        }
        int64_t h2 = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(1);
        if ((h2 <= NUM_ZERO) || (h2 > MAX_SHARED_H_SHAPE_SIZE)) {
            OP_LOGE(A5_INNER_DEBUG, "h2 should be in (0, 12288], but got %lu!", h2);
            return false;
        }
        int64_t n2 = tilingData.commonTilingInfo.get_isMmWeightTrans() ?
                          context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(0) :
                          context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
        if ((n2 <= NUM_ZERO) || (n2 >= MAX_SHAPE_SIZE)) {
            OP_LOGE(A5_INNER_DEBUG, "n2 should be in (0, 65536), but got %lu!", n2);
            return false;
        }
        int64_t topK = bsK / bs;
        if ((topK < NUM_TWO) || (topK > NUM_EIGHT)) {
            OP_LOGE(A5_INNER_DEBUG, "topK should be in [2, 8], but got %lu!", topK);
            return false;
        }
    }
    return true;
}

static bool CheckDimValue(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    auto attrs = context->GetAttrs();
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    auto gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    auto y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);

    int64_t bsK = gmmX->GetStorageShape().GetDim(0);
    int64_t h = gmmX->GetStorageShape().GetDim(1);
    int64_t eOverEp = gmmWeight->GetStorageShape().GetDim(0);
    int64_t gmmWeightH = tilingData.commonTilingInfo.get_isGmmWeightTrans() ? gmmWeight->GetStorageShape().GetDim(2) :
                                                                              gmmWeight->GetStorageShape().GetDim(1);
    int64_t n1 = tilingData.commonTilingInfo.get_isGmmWeightTrans() ? gmmWeight->GetStorageShape().GetDim(1) :
                                                                      gmmWeight->GetStorageShape().GetDim(2);
    int64_t a = y->GetStorageShape().GetDim(0);
    int64_t yDim1 = y->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(
        !CheckDimValueShape(context, tilingData), OP_LOGE(A5_INNER_DEBUG, "CheckDimValueShape failed."),
        return false);

    OP_TILING_CHECK(
        !CheckDimValueIsNeedMM(context, tilingData), OP_LOGE(A5_INNER_DEBUG, "CheckDimValueIsNeedMM failed."),
        return false);

    OP_TILING_CHECK(
        h != gmmWeightH, OP_LOGE(A5_INNER_DEBUG, "gmmWeightH[%ld] should equal to h[%ld].", gmmWeightH, h),
        return false);
    OP_TILING_CHECK(
        n1 != yDim1, OP_LOGE(A5_INNER_DEBUG, "n1[%ld] should equal to yDim1[%ld].", n1, yDim1), return false);
    OP_TILING_CHECK(
        h > MAX_MATMUL_K,
        OP_LOGE(A5_INNER_DEBUG, "h[%ld] exceeds matmul's limit[%ld]", h, MAX_MATMUL_K), return false);
    OP_TILING_CHECK(eOverEp > MAX_E,
    OP_LOGE(A5_INNER_DEBUG, "eOverEp[%ld] exceeds limit[%ld]", eOverEp, MAX_E), return false);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    size_t sendSize = sendCountsPtr->GetSize();
    OP_TILING_CHECK(
        sendSize != eOverEp * tilingData.commonTilingInfo.get_epWorldSize(),
        OP_LOGE(
            A5_INNER_DEBUG, "sendCounts size[%ld] doesnot equal to expert num[%ld]", sendSize,
            eOverEp * tilingData.commonTilingInfo.get_epWorldSize()),
        return false);
    const int64_t* sendArray = static_cast<const int64_t*>(sendCountsPtr->GetData());
    int64_t sendCountSum = 0;
    for (size_t i = 0; i < sendSize; i++) {
        sendCountSum += sendArray[i];
    }
    OP_TILING_CHECK(
        sendCountSum != bsK,
        OP_LOGE(A5_INNER_DEBUG, "bsK[%ld] should be equal to sum of sendCounts[%ld]", bsK, sendCountSum), return false);

    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    size_t recvSize = recvCountsPtr->GetSize();
    OP_TILING_CHECK(
        recvSize != eOverEp * tilingData.commonTilingInfo.get_epWorldSize(),
        OP_LOGE(
            A5_INNER_DEBUG, "recvCounts size[%ld] doesnot equal to expert num[%ld]", recvSize,
            eOverEp * tilingData.commonTilingInfo.get_epWorldSize()),
        return false);
    const int64_t* recvArray = static_cast<const int64_t*>(recvCountsPtr->GetData());
    int64_t recvCountSum = 0;
    for (size_t i = 0; i < recvSize; i++) {
        recvCountSum += recvArray[i];
    }
    OP_TILING_CHECK(
        recvCountSum != a, OP_LOGE(A5_INNER_DEBUG, "a[%ld] should be equal to sum of recvCounts[%ld]", a, recvCountSum),
        return false);

    OP_TILING_CHECK(
        !CheckSharedExpDimValue(context, tilingData), OP_LOGE(A5_INNER_DEBUG, "CheckSharedExpDimValue failed"),
        return false);
    return true;
}

static bool CheckInputAndOutput(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    auto attrs = context->GetAttrs();

    const gert::StorageShape* gmmX = context->GetInputShape(GMM_X_INDEX);
    const gert::StorageShape* gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    const gert::StorageShape* sendCounts = context->GetOptionalInputShape(SEND_COUNTS_TENSOR_INDEX);
    const gert::StorageShape* recvCounts = context->GetOptionalInputShape(RECV_COUNTS_TENSOR_INDEX);
    const gert::StorageShape* mmWeight = context->GetOptionalInputShape(MM_WEIGHT_INDEX);
    const gert::StorageShape* y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);
    const gert::StorageShape* mmY = context->GetOutputShape(OUTPUT_MM_Y_INDEX);

    OP_TILING_CHECK(gmmX == nullptr, OP_LOGE(A5_INNER_DEBUG, "GmmX is nullptr!"), return false);
    OP_TILING_CHECK(gmmWeight == nullptr, OP_LOGE(A5_INNER_DEBUG, "GmmWeight is nullptr!"), return false);
    OP_TILING_CHECK(y == nullptr, OP_LOGE(A5_INNER_DEBUG, "GmmY is nullptr!"), return false);

    if ((sendCounts != nullptr) || (recvCounts != nullptr)) {
        OP_LOGW(A5_INNER_DEBUG, "sendCountsTensorOptional and recvCountsTensorOptional should be nullptr.");
    }
    tilingData.commonTilingInfo.set_isSendCntsTensor(
        context->GetOptionalInputShape(SEND_COUNTS_TENSOR_INDEX) != nullptr);
    tilingData.commonTilingInfo.set_isRecvCntsTensor(
        context->GetOptionalInputShape(RECV_COUNTS_TENSOR_INDEX) != nullptr);

    auto permuteOutFlagPtr = attrs->GetAttrPointer<bool>(ATTR_PERMUTE_OUT_FLAG_INDEX);
    if (permuteOutFlagPtr != nullptr) {
        tilingData.commonTilingInfo.set_isPermuteOut(*permuteOutFlagPtr);
    }

    if (context->GetOptionalInputShape(MM_X_INDEX) != nullptr) {
        OP_TILING_CHECK(
            mmWeight == nullptr, OP_LOGE(A5_INNER_DEBUG, "mmWeight is nullptr when mmX is given!"), return false);
        OP_TILING_CHECK(mmY == nullptr, OP_LOGE(A5_INNER_DEBUG, "mmY is nullptr when mmX is given!"), return false);
    }
    tilingData.commonTilingInfo.set_isNeedMM(context->GetOptionalInputShape(MM_X_INDEX) != nullptr);

    OP_TILING_CHECK(!CheckDType(context, tilingData), OP_LOGE(A5_INNER_DEBUG, "CheckDType failed!"), return false);
    OP_TILING_CHECK(!CheckFormat(context, tilingData), OP_LOGE(A5_INNER_DEBUG, "CheckFormat failed!"), return false);
    OP_TILING_CHECK(!CheckDimNum(context, tilingData), OP_LOGE(A5_INNER_DEBUG, "CheckDimNum failed!"), return false);
    OP_TILING_CHECK(!CheckDimValue(context, tilingData), OP_LOGE(A5_INNER_DEBUG, "CheckDimValue failed!"), return false);
    return true;
}

static ge::graphStatus GetContextAttr(
    const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A5_INNER_DEBUG, "GetAttrs returned nullptr!"), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_EP_WORLD_SIZE_INDEX);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    auto permuteOutFlagPtr = attrs->GetAttrPointer<bool>(ATTR_PERMUTE_OUT_FLAG_INDEX);

    OP_TILING_CHECK(
        groupEpPtr == nullptr, OP_LOGE(A5_INNER_DEBUG, "groupEpPtr returned nullptr!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        epWorldSizePtr == nullptr, OP_LOGE(A5_INNER_DEBUG, "epWorldSizePtr returned nullptr!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        sendCountsPtr == nullptr, OP_LOGE(A5_INNER_DEBUG, "sendCountsPtr returned nullptr!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        recvCountsPtr == nullptr, OP_LOGE(A5_INNER_DEBUG, "recvCountsPtr returned nullptr!"), return ge::GRAPH_FAILED);

    tilingData.commonTilingInfo.set_epWorldSize(*epWorldSizePtr);
    tilingData.commonTilingInfo.set_isGmmWeightTrans(*transGmmWeightPtr);
    tilingData.commonTilingInfo.set_isMmWeightTrans(*transMmWeightPtr);
    tilingData.commonTilingInfo.set_isPermuteOut(*permuteOutFlagPtr);

    std::copy_n(
        static_cast<const int64_t*>(recvCountsPtr->GetData()), recvCountsPtr->GetSize(),
        tilingData.aicpuTiling.get_recvCnt());
    std::copy_n(
        static_cast<const int64_t*>(sendCountsPtr->GetData()), sendCountsPtr->GetSize(),
        tilingData.aicpuTiling.get_sendCnt());

    OP_LOGI(
        A5_INNER_DEBUG, "epGroup is %s, epWorldSize is %lu", groupEpPtr, tilingData.commonTilingInfo.get_epWorldSize());
    return ge::GRAPH_SUCCESS;
}

static bool GetShape(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    OP_TILING_CHECK(
        (context->GetInputShape(GMM_X_INDEX) == nullptr), 
        OP_LOGE(
            A5_INNER_DEBUG, "gmm_x is nullptr."),
        return false);
    OP_TILING_CHECK(
        (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr), 
        OP_LOGE(
            A5_INNER_DEBUG, "gmm_weight is nullptr."),
        return false);
    OP_TILING_CHECK(
        (context->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr), 
        OP_LOGE(
            A5_INNER_DEBUG, "gmm_y is nullptr."),
        return false);

    tilingData.commonTilingInfo.set_BSK(context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0));
    tilingData.commonTilingInfo.set_H1(context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1));
    tilingData.commonTilingInfo.set_E_ep(context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(0));
    tilingData.commonTilingInfo.set_N1(
        tilingData.commonTilingInfo.get_isGmmWeightTrans() ?
            context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1) :
            context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_TWO));

    tilingData.commonTilingInfo.set_A(context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(0));

    auto mmDtype = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    auto mmDataTypeSize = GetSizeByDataType(mmDtype);

    if (tilingData.commonTilingInfo.get_isNeedMM()) {
        tilingData.commonTilingInfo.set_BS(context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(0));
        tilingData.commonTilingInfo.set_H2(context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(1));
        tilingData.commonTilingInfo.set_N2(
            tilingData.commonTilingInfo.get_isMmWeightTrans() ?
                context->GetInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(0) :
                context->GetInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(1));
    } else {
        tilingData.commonTilingInfo.set_BS(0);
        tilingData.commonTilingInfo.set_H2(0);
        tilingData.commonTilingInfo.set_N2(0);
    }

    OP_LOGD(
        A5_INNER_DEBUG,
        "commonTiling Info: Bsk %lu, H1 %lu, eOverEp %lu, A %lu, BS %lu, H2%lu, N2%lu, isNeedMM %u, mmDtype %d, "
        "mmDataTypeSize %d",
        tilingData.commonTilingInfo.get_BSK(), tilingData.commonTilingInfo.get_H1(),
        tilingData.commonTilingInfo.get_E_ep(), tilingData.commonTilingInfo.get_A(),
        tilingData.commonTilingInfo.get_BS(), tilingData.commonTilingInfo.get_H2(),
        tilingData.commonTilingInfo.get_N2(), tilingData.commonTilingInfo.get_isNeedMM() ? 1U : 0U,
        static_cast<int>(mmDtype), mmDataTypeSize);
    return true;
}

static ge::graphStatus CheckMKN(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    auto mmDtype = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    auto mmDataTypeSize = GetSizeByDataType(mmDtype);

    // NOTE: In trunk_ai ge::TypeUtils::DataTypeToAscendString(xx).GetString() is used
    OP_TILING_CHECK(
        mmDataTypeSize == 0,
        OP_LOGE(
            A5_INNER_DEBUG, "GMM get matmul dtype[%s] size is 0.", TypeUtils::DataTypeToSerialString(mmDtype).c_str()),
        return ge::GRAPH_FAILED);

    uint32_t numInOneBlk = ONE_BLK_SIZE / mmDataTypeSize;
    OP_TILING_CHECK(numInOneBlk == 0, OP_LOGE(A5_INNER_DEBUG, "GMM numInOneBlk cannot be 0."), return ge::GRAPH_FAILED);

    int64_t maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
    uint32_t maxMKNuint = static_cast<uint32_t>(maxMKN);
    uint32_t maxM = tilingData.commonTilingInfo.get_A();
    uint32_t maxK = tilingData.commonTilingInfo.get_H1();
    uint32_t maxN = tilingData.commonTilingInfo.get_N1();
    OP_TILING_CHECK(
        ((maxM > maxMKNuint) || (maxK > maxMKNuint) || (maxN > maxMKNuint)),
        OP_LOGE(
            A5_INNER_DEBUG, "32B-aligned m[%u], k[%u] or n[%u] axis for gmm is out of range int32", maxM, maxK, maxN),
        return ge::GRAPH_FAILED);

    if (tilingData.commonTilingInfo.get_isNeedMM()) {
        uint32_t maxMForMM = tilingData.commonTilingInfo.get_BS();
        uint32_t maxKForMM = tilingData.commonTilingInfo.get_H2();
        uint32_t maxNForMM = tilingData.commonTilingInfo.get_N2();
        OP_TILING_CHECK(
            ((maxMForMM > maxMKNuint) || (maxKForMM > maxMKNuint) || (maxNForMM > maxMKNuint)),
            OP_LOGE(
                A5_INNER_DEBUG, "32B-aligned m[%u], k[%u] or n[%u] axis for mm is out of range int32", maxMForMM,
                maxKForMM, maxNForMM),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static void SetHcclTiling(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    tilingData.set_version(HCCL_VERSION);
    tilingData.set_hcommCnt(1);

    tilingData.hcommCfgATA.set_srcDataType(static_cast<uint32_t>(
        mc2tiling::ConvertGeTypeToHcclType(A5_INNER_DEBUG, context->GetInputDesc(GMM_X_INDEX)->GetDataType())));
    tilingData.hcommCfgATA.set_dstDataType(static_cast<uint32_t>(
        mc2tiling::ConvertGeTypeToHcclType(A5_INNER_DEBUG, context->GetInputDesc(GMM_X_INDEX)->GetDataType())));
    tilingData.hcommCfgATA.set_opType(HCCL_CMD_ALLTOALLV);
}

static uint64_t GetTilingKeyA5(const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    uint64_t tilingKey = 0UL;
    tilingKey = (context->GetInputDesc(GMM_X_INDEX)->GetDataType() == ge::DT_FLOAT16) ? TILINGKEY_FP16 : TILINGKEY_BF16;
    tilingKey += (tilingData.commonTilingInfo.get_isGmmWeightTrans()) ? TILINGKEY_GMM_WEIGHT_TRANSPOSE : 0UL;
    tilingKey += (tilingData.commonTilingInfo.get_isNeedMM()) ? TILINGKEY_MM : 0UL;
    tilingKey += TILINGKEY_A5_OFFSET;
    if (tilingData.commonTilingInfo.get_isNeedMM()) {
        tilingKey += (tilingData.commonTilingInfo.get_isMmWeightTrans()) ? TILINGKEY_MM_WEIGHT_TRANSPOSE : 0UL;
    }
    OP_LOGD(A5_INNER_DEBUG, "tilingKey is %lu", tilingKey);
    return tilingKey;
}

static ge::graphStatus SetMMTiling(
    const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData, uint32_t curBaseM,
    uint32_t curBaseN, bool isSharedExpert)
{
    OP_LOGD(A5_INNER_DEBUG, "Begin SetMMTiling");

    auto matmulDType = matmul_tiling::DataType::DT_FLOAT16;
    if (context->GetInputDesc(GMM_X_INDEX)->GetDataType() == ge::DT_BF16) {
        matmulDType = matmul_tiling::DataType::DT_BF16;
    }
    OP_LOGD(
        A5_INNER_DEBUG, "mmDtype is %d, dTypeforMM is %d",
        static_cast<int>(context->GetInputDesc(GMM_X_INDEX)->GetDataType()), static_cast<int>(matmulDType));

    uint32_t maxM = !isSharedExpert ? tilingData.commonTilingInfo.get_A() : tilingData.commonTilingInfo.get_BS();
    uint32_t maxK = !isSharedExpert ? tilingData.commonTilingInfo.get_H1() : tilingData.commonTilingInfo.get_H2();
    uint32_t maxN = !isSharedExpert ? tilingData.commonTilingInfo.get_N1() : tilingData.commonTilingInfo.get_N2();

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);
    uint64_t ubSize = PLATFORM_SIZE.ubSize;

    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDType, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDType, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, matmulDType);
    mm.SetOrgShape(maxM, maxN, maxK);
    mm.SetShape(maxM, maxN, maxK);
    mm.SetFixSplit(std::min(curBaseM, maxM), curBaseN);
    mm.SetBufferSpace(PLATFORM_SIZE.l1Size, PLATFORM_SIZE.l0CSize, ubSize);
    OP_LOGD(
        A5_INNER_DEBUG,
        "gmm matmul gettiling start, maxM %u, maxK %u, maxN %u, curBaseM %u, curBaseN %u, l1Size %lu, l0CSize %lu, "
        "ubSize %lu.",
        maxM, maxK, maxN, curBaseM, curBaseN, PLATFORM_SIZE.l1Size, PLATFORM_SIZE.l0CSize, ubSize);
    if (!isSharedExpert) {
        OP_TILING_CHECK(
            mm.GetTiling(tilingData.gmmTilingData) == -1, OP_LOGE(A5_INNER_DEBUG, "gmm matmul getTiling failed."),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            mm.GetTiling(tilingData.mmTilingData) == -1, OP_LOGE(A5_INNER_DEBUG, "mm matmul getTiling failed."),
            return ge::GRAPH_FAILED);
    }
    OP_LOGD(A5_INNER_DEBUG, "Finish SetMMTiling");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalAndSetMMTiling(
    const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData, bool isSharedExpert)
{
    OP_LOGD(A5_INNER_DEBUG, "Begin CalAndSetMMTiling");

    uint32_t maxM = !isSharedExpert ? tilingData.commonTilingInfo.get_A() : tilingData.commonTilingInfo.get_BS();
    uint32_t maxK = !isSharedExpert ? tilingData.commonTilingInfo.get_H1() : tilingData.commonTilingInfo.get_H2();
    uint32_t maxN = !isSharedExpert ? tilingData.commonTilingInfo.get_N1() : tilingData.commonTilingInfo.get_N2();

    uint32_t mmDataTypeSize = GetSizeByDataType(context->GetInputDesc(GMM_X_INDEX)->GetDataType());

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);

    uint32_t tempBaseN = BEST_BASE_N;
    while (tempBaseN > maxN) {
        tempBaseN = tempBaseN >> 1;
    }
    if (tempBaseN < maxN) {
        tempBaseN = tempBaseN << 1;
    }
    uint32_t curBaseN = std::min<int32_t>(BEST_BASE_N, tempBaseN);
    OP_TILING_CHECK(curBaseN == 0, OP_LOGE(A5_INNER_DEBUG, "curBaseN cannot be 0."), return ge::GRAPH_FAILED);

    uint32_t curBaseK = (PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B) / (curBaseN * mmDataTypeSize);
    curBaseK = SixteenAlign(curBaseK);
    if (curBaseK > MAX_BASE_K) {
        curBaseK = MAX_BASE_K;
        int32_t maxBaseN = SixteenAlign(PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B / (curBaseK * mmDataTypeSize));
        curBaseN = std::min<int32_t>(curBaseN, maxBaseN);
        curBaseN = std::max<int32_t>(16, SixteenAlign(curBaseN, true)); // 16: min value for baseN
    }
    if (curBaseK > maxK) {
        curBaseK = std::min<int32_t>(curBaseK, SixteenAlign(maxK, true));
    }
    OP_TILING_CHECK(curBaseK == 0, OP_LOGE(A5_INNER_DEBUG, "curBaseK cannot be 0."), return ge::GRAPH_FAILED);

    // 基于使能 double buffer的L0A内存与L0B内存计算BaseM(cube)
    uint32_t maxBaseM = PLATFORM_SIZE.l0CSize / (curBaseN * mmDataTypeSize);
    uint32_t curBaseM =
        std::min<uint32_t>((PLATFORM_SIZE.l0ASize / DOUBLE_BUFFER_L0A_L0B / (curBaseK * mmDataTypeSize)), maxBaseM);
    curBaseM = SixteenAlign(curBaseM);
    if (curBaseM > maxM) {
        curBaseM = SixteenAlign(maxM, true);
    }
    OP_TILING_CHECK(curBaseM == 0, OP_LOGE(A5_INNER_DEBUG, "curBaseM cannot be 0."), return ge::GRAPH_FAILED);
    OP_LOGD(A5_INNER_DEBUG, "curBaseM=%du, curBaseK=%u, curBaseN=%u.", curBaseM, curBaseK, curBaseN);
    OP_LOGD(A5_INNER_DEBUG, "Finish CalAndSetMMTiling");
    return SetMMTiling(context, tilingData, curBaseM, curBaseN, isSharedExpert);
}

static ge::graphStatus DoAiCoreTiling(
    const gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    OP_LOGD(A5_INNER_DEBUG, "Begin DoAiCoreTiling");

    OP_TILING_CHECK(
        CalAndSetMMTiling(context, tilingData, false) != ge::GRAPH_SUCCESS,
        OP_LOGE(A5_INNER_DEBUG, "GMM CalMMTiling failed."), return ge::GRAPH_FAILED);
    if (tilingData.commonTilingInfo.get_isNeedMM()) {
        OP_TILING_CHECK(
            CalAndSetMMTiling(context, tilingData, true) != ge::GRAPH_SUCCESS,
            OP_LOGE(A5_INNER_DEBUG, "MM CalMMTiling failed."), return ge::GRAPH_FAILED);
    }
    OP_LOGD(A5_INNER_DEBUG, "Finish DoAiCoreTiling");
    return ge::GRAPH_SUCCESS;
}

static void SetTilingData(gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    OP_TILING_CHECK(context->GetRawTilingData() == nullptr, OP_LOGE(A5_INNER_DEBUG, "tiling data is nullptr."), return);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static bool CheckIsNeedMM(gert::TilingContext* context, AlltoAllvGroupedMatMulTilingDataA5& tilingData)
{
    if (!tilingData.commonTilingInfo.get_isNeedMM()) {
            OP_TILING_CHECK(
            (context->GetOutputShape(OUTPUT_MM_Y_INDEX) != nullptr &&
             context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum() != NUM_ZERO),
            OP_LOGE(A5_INNER_DEBUG, "The mmY should be null when mmX and mmWeight are null!"), return false);

        if (tilingData.commonTilingInfo.get_isMmWeightTrans()) {
            OP_LOGE(A5_INNER_DEBUG, "The trans_mm_weight should be false when mmX mmWeight mmY is null!");
            return false;
        }
    }
    return true;
}

static ge::graphStatus RunFusionKernelTilingA5(gert::TilingContext* context)
{
    OP_LOGD(A5_INNER_DEBUG, "Begin RunFusionKernelTiling");
    AlltoAllvGroupedMatMulTilingDataA5 tilingData;

    // 设置 CV 核数
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);
    tilingData.commonTilingInfo.set_aivCoreNum(aivNum);
    tilingData.commonTilingInfo.set_aicCoreNum(aicNum);

    OP_TILING_CHECK(
        context->GetInputDesc(GMM_X_INDEX) == nullptr, OP_LOGE(A5_INNER_DEBUG, "the input desc of gmm_x is nullptr."),
        return ge::GRAPH_FAILED);
    SetHcclTiling(context, tilingData);
    OP_TILING_CHECK(
        GetContextAttr(context, tilingData) != ge::GRAPH_SUCCESS, OP_LOGE(A5_INNER_DEBUG, "GetContextAttr failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !CheckInputAndOutput(context, tilingData), OP_LOGE(A5_INNER_DEBUG, "CheckInputAndOutput failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !GetShape(context, tilingData), OP_LOGE(A5_INNER_DEBUG, "GetShape failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckMKN(context, tilingData) != ge::GRAPH_SUCCESS, OP_LOGE(A5_INNER_DEBUG, "CheckMKN failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        DoAiCoreTiling(context, tilingData) != ge::GRAPH_SUCCESS, OP_LOGE(A5_INNER_DEBUG, "DoAiCoreTiling failed."),
        return ge::GRAPH_FAILED);
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(
        workspaceSize == nullptr, OP_LOGE(A5_INNER_DEBUG, "workspace is nullptr."), return ge::GRAPH_FAILED);
    uint64_t commOut = tilingData.commonTilingInfo.get_A() * tilingData.commonTilingInfo.get_H1() * FP16_DATASIZE;
    uint64_t permuteOut = tilingData.commonTilingInfo.get_isPermuteOut() ? 0 :
            (tilingData.commonTilingInfo.get_A() * tilingData.commonTilingInfo.get_H1() * FP16_DATASIZE);
    workspaceSize[0] = SYS_WORKSPACE_SIZE + commOut + permuteOut;
    OP_LOGD(
        A5_INNER_DEBUG, "commOut %lu, permuteOut %lu, workspace %lu", commOut, permuteOut,
        static_cast<uint64_t>(workspaceSize[0]));
    if (!CheckIsNeedMM(context, tilingData)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tilingKey = GetTilingKeyA5(context, tilingData);
    context->SetTilingKey(tilingKey);
    PrintTilingData(tilingData);
    SetTilingData(context, tilingData);
    OP_LOGD(A5_INNER_DEBUG, "Finish RunFusionKernelTiling");
    return ge::GRAPH_SUCCESS;
}

bool AlltoAllvGmmTilingA5::IsCapable()
{
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND910_95) {
        OP_LOGD(A5_INNER_DEBUG, "Do AlltoAllvGmmTilingA5 tiling.");
        return true;
    }
    return false;
}

ge::graphStatus AlltoAllvGmmTilingA5::DoOpTiling()
{
    return RunFusionKernelTilingA5(context_);
}

uint64_t AlltoAllvGmmTilingA5::GetTilingKey() const
{
    // tilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(A5_INNER_DEBUG, "AlltoAllvGmmTilingA5 get tiling key %lu", tilingKey);
    return tilingKey;
}
}; // namespace optiling
