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
 * \file grouped_mat_mul_allto_allv_tiling_A5.cc
 * \brief
 */

#include "grouped_mat_mul_allto_allv_tiling_A5.h"

#include <string>
#include <numeric>
#include <vector>
#include "mc2_hcom_topo_info.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "tiling/mc2_tiling_utils.h"
#include "tiling/mc2_tiling_common_var.h"
#include "../grouped_mat_mul_allto_allv_tiling_base.h"
#include "mc2_log.h"


using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

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

constexpr uint32_t DIM_TWO = 2U;
constexpr uint64_t DIM_ONE = 1UL;
constexpr uint64_t DIM_THREE = 3UL;

constexpr uint32_t HCCL_CMD_ALLGATHER = 6U;
constexpr uint32_t HCCL_CMD_ALLTOALLV = 8U;
constexpr uint32_t HCCL_VERSION = 3U;

constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_EIGHT = 8;
constexpr int64_t MAX_DIM_VALUE = 65536;
constexpr int64_t MAX_BSK_VALUE = 52428800;
constexpr uint32_t MAX_SHARED_H_SHAPE_SIZE = 12288;
constexpr int64_t MAX_EXPERT_NUM_PER_RANK = 32;

constexpr uint32_t ATTR_GROUP_INDEX = 0U;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1U;
constexpr uint32_t ATTR_SEND_COUNTS_INDEX = 2U;
constexpr uint32_t ATTR_RECV_COUNTS_INDEX = 3U;
constexpr uint32_t ATTR_TRANS_GMM_WEIGHT_INDEX = 4U;
constexpr uint32_t ATTR_TRANS_MM_WEIGHT_INDEX = 5U;

constexpr uint64_t TILINGKEY_COMPUTE_OPTIONAL_MM = 1UL;
constexpr uint64_t TILINGKEY_GMM_WEIGHT_TRANSPOSE = 10UL;
constexpr uint64_t TILINGKEY_MM_WEIGHT_TRANSPOSE = 100UL;
constexpr uint64_t TILINGKEY_A5_OFFSET = 1000000000000000000UL;

constexpr int64_t MAX_E = 128U;
constexpr int64_t MAX_MATMUL_K = 65535U;

constexpr int64_t BEST_L1_PARTA = 256L * 1024L;
constexpr int64_t BEST_L1_PARTB = 128L * 1024L;
constexpr int64_t BEST_BASE_N = 256L;
constexpr uint32_t UB_DEVIDE_NUM = 2U;
constexpr uint32_t UB_CALSIZE_PER_BLOCK = 16U * 1024U;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2UL;
constexpr uint64_t DOUBEL_BUFFER_STEPKA_STEPKB = 2UL;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;
constexpr uint32_t MAX_TURN_NUM = 24U;
constexpr uint32_t MAX_BASE_K = 128;
constexpr uint64_t COMM_TILE = 8; // 每卡数据分配几次计算

const char* C5_INNER_DEBUG = "GroupedMatmulAlltoAllv Tiling";

static uint32_t maxM = 0;
static uint32_t maxN = 0;
static uint32_t maxK = 0;
static uint32_t baseM_ = 0;
static uint32_t baseN_ = 0;
static uint32_t baseK_ = 0;

static uint64_t GMMGetSizePlatForm(
    const platform_ascendc::CoreMemType memType, platform_ascendc::PlatformAscendC ascendcPlatform)
{
    uint64_t size = 0UL;
    ascendcPlatform.GetCoreMemSize(memType, size);
    return size;
}

static void PrintTilingData(GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    OP_LOGD(C5_INNER_DEBUG, "version %u", tilingData.get_version());
    OP_LOGD(C5_INNER_DEBUG, "hcommCnt %u", tilingData.get_hcommCnt());

    OP_LOGD(C5_INNER_DEBUG, "commonTilingInfo A %lu", tilingData.commonTilingInfo.get_A());
    OP_LOGD(C5_INNER_DEBUG, "commonTilingInfo sharedMatmulH %lu", tilingData.commonTilingInfo.get_sharedMatmulH());
    OP_LOGD(C5_INNER_DEBUG, "commonTilingInfo E_ep %lu", tilingData.commonTilingInfo.get_E_ep());
    OP_LOGD(C5_INNER_DEBUG, "commonTilingInfo N1 %lu", tilingData.commonTilingInfo.get_N1());
    OP_LOGD(C5_INNER_DEBUG, "commonTilingInfo BS %lu", tilingData.commonTilingInfo.get_Bs());
    OP_LOGD(C5_INNER_DEBUG, "commonTilingInfo N2 %lu", tilingData.commonTilingInfo.get_N2());
    OP_LOGD(C5_INNER_DEBUG, "commonTilingInfo BsK %lu", tilingData.commonTilingInfo.get_BsK());
    OP_LOGD(C5_INNER_DEBUG, "commonTilingInfo epWorldSize %lu", tilingData.commonTilingInfo.get_epWorldSize());
    OP_LOGD(C5_INNER_DEBUG, "commonTilingInfo aivCoreNum %lu", tilingData.commonTilingInfo.get_aivCoreNum());
    OP_LOGD(C5_INNER_DEBUG, "commonTilingInfo aicCoreNum %lu", tilingData.commonTilingInfo.get_aicCoreNum());
    OP_LOGD(
        C5_INNER_DEBUG, "commonTilingInfo isGmmWeightTrans %u",
        tilingData.commonTilingInfo.get_isGmmWeightTrans() ? 1U : 0U);
    OP_LOGD(
        C5_INNER_DEBUG, "commonTilingInfo isMmWeightTrans %u",
        tilingData.commonTilingInfo.get_isMmWeightTrans() ? 1U : 0U);
    OP_LOGD(
        C5_INNER_DEBUG, "commonTilingInfo isOptionalMatmul %u",
        tilingData.commonTilingInfo.get_isOptionalMatmul() ? 1U : 0U);
    OP_LOGD(
        C5_INNER_DEBUG, "commonTilingInfo isOptionaSendRecvCountTensors %u",
        tilingData.commonTilingInfo.get_isOptionaSendRecvCountTensors() ? 1U : 0U);

    Mc2Log::PrintTCubeTilingData(C5_INNER_DEBUG, tilingData.matmulTiling);
    if (tilingData.commonTilingInfo.get_isOptionalMatmul()) {
        OP_LOGD(C5_INNER_DEBUG, "Has shared expoert.");
        Mc2Log::PrintTCubeTilingData(C5_INNER_DEBUG, tilingData.sharedExpMatmulTiling);
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

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15U; // 15: 16 bytes up-align
    }
    return a & ~15U; // ~15: 16bytes down-align
}

static bool CheckDType(const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto gmmXDType = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    auto gmmWeightDType = context->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    auto yDType = context->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetDataType();

    OP_TILING_CHECK(
        !((gmmXDType == ge::DT_FLOAT16) || (gmmXDType == ge::DT_BF16)),
        OP_LOGE(
            C5_INNER_DEBUG, "GmmX DataType [%s] is not supported.",
            TypeUtils::DataTypeToSerialString(gmmXDType).c_str()),
        return false);
    OP_TILING_CHECK(
        !((gmmWeightDType == ge::DT_FLOAT16) || (gmmWeightDType == ge::DT_BF16)),
        OP_LOGE(
            C5_INNER_DEBUG, "GmmWeight DataType [%s] is not supported.",
            TypeUtils::DataTypeToSerialString(gmmWeightDType).c_str()),
        return false);
    OP_TILING_CHECK(
        !((yDType == ge::DT_FLOAT16) || (yDType == ge::DT_BF16)),
        OP_LOGE(
            C5_INNER_DEBUG, "GmmY DataType [%s] is not supported.", TypeUtils::DataTypeToSerialString(yDType).c_str()),
        return false);
    OP_TILING_CHECK(
        !((gmmXDType == gmmWeightDType) && (gmmXDType == yDType)),
        OP_LOGE(
            C5_INNER_DEBUG, "GmmX [%s], GmmWeight [%s], GmmY [%s] should be same.",
            TypeUtils::DataTypeToSerialString(gmmXDType).c_str(),
            TypeUtils::DataTypeToSerialString(gmmWeightDType).c_str(),
            TypeUtils::DataTypeToSerialString(yDType).c_str()),
        return false);

    if (tilingData.commonTilingInfo.get_isOptionaSendRecvCountTensors()) {
        auto sendCntsDType = context->GetOptionalInputDesc(SEND_COUNTS_TENSOR_INDEX)->GetDataType();
        OP_TILING_CHECK(
            !((sendCntsDType == ge::DT_INT32) || (sendCntsDType == ge::DT_INT64)),
            OP_LOGE(
                C5_INNER_DEBUG, "SendCountsTensor DataType [%s] is not supported.",
                TypeUtils::DataTypeToSerialString(sendCntsDType).c_str()),
            return false);
            
        auto recvCntsDType = context->GetOptionalInputDesc(RECV_COUNTS_TENSOR_INDEX)->GetDataType();
        OP_TILING_CHECK(
            !((recvCntsDType == ge::DT_INT32) || (recvCntsDType == ge::DT_INT64)),
            OP_LOGE(
                C5_INNER_DEBUG, "RecvCountsTensor DataType [%s] is not supported.",
                TypeUtils::DataTypeToSerialString(recvCntsDType).c_str()),
            return false);
        OP_TILING_CHECK(
            !((sendCntsDType == recvCntsDType)),
            OP_LOGE(
                C5_INNER_DEBUG, "SendCountsTensor [%s], RecvCountsTensor [%s] should be same.",
                TypeUtils::DataTypeToSerialString(sendCntsDType).c_str(),
                TypeUtils::DataTypeToSerialString(recvCntsDType).c_str()),
            return false);
    }

    if (tilingData.commonTilingInfo.get_isOptionalMatmul()) {
        auto mmXDType = context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType();
        auto mmWeightDType = context->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType();
        auto mmYDType = context->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType();
        OP_TILING_CHECK(
            !((gmmXDType == mmXDType) && (gmmXDType == mmWeightDType) && (gmmXDType == mmYDType)),
            OP_LOGE(
                C5_INNER_DEBUG, "MmX [%s], MmWeight [%s], MmY [%s] should be same as GMM [%s].",
                TypeUtils::DataTypeToSerialString(mmXDType).c_str(),
                TypeUtils::DataTypeToSerialString(mmWeightDType).c_str(),
                TypeUtils::DataTypeToSerialString(mmYDType).c_str(),
                TypeUtils::DataTypeToSerialString(gmmXDType).c_str()),
            return false);
    }

    return true;
}

static bool CheckFormat(const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto gmmXFormat = context->GetInputDesc(GMM_X_INDEX)->GetFormat().GetStorageFormat();
    auto gmmWeightFormat = context->GetInputDesc(GMM_WEIGHT_INDEX)->GetFormat().GetStorageFormat();
    auto yFormat = context->GetOutputDesc(OUTPUT_GMM_Y_INDEX)->GetFormat().GetStorageFormat();

    OP_TILING_CHECK(!(gmmXFormat == ge::FORMAT_ND), OP_LOGE(C5_INNER_DEBUG, "GmmX Format should be ND."), return false);
    OP_TILING_CHECK(
        !(gmmWeightFormat == ge::FORMAT_ND), OP_LOGE(C5_INNER_DEBUG, "GmmWeight Format should be ND."), return false);
    OP_TILING_CHECK(!(yFormat == ge::FORMAT_ND), OP_LOGE(C5_INNER_DEBUG, "GmmY Format should be ND."), return false);

    if (tilingData.commonTilingInfo.get_isOptionaSendRecvCountTensors()) {
        auto recvCntsFormat = context->GetOptionalInputDesc(RECV_COUNTS_TENSOR_INDEX)->GetFormat().GetStorageFormat();
        OP_TILING_CHECK(
            !(recvCntsFormat == ge::FORMAT_ND), OP_LOGE(C5_INNER_DEBUG, "RecvCountsTensor Format should be ND."),
            return false);

        auto sendCntsFormat = context->GetOptionalInputDesc(SEND_COUNTS_TENSOR_INDEX)->GetFormat().GetStorageFormat();
        OP_TILING_CHECK(
            !(sendCntsFormat == ge::FORMAT_ND), OP_LOGE(C5_INNER_DEBUG, "SendCountsTensor Format should be ND."),
            return false);
    }

    if (tilingData.commonTilingInfo.get_isOptionalMatmul()) {
        auto mmXFormat = context->GetOptionalInputDesc(MM_X_INDEX)->GetFormat().GetStorageFormat();
        auto mmWeightFormat = context->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetFormat().GetStorageFormat();
        auto mmYFormat = context->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetFormat().GetStorageFormat();
        OP_TILING_CHECK(
            !(mmXFormat == ge::FORMAT_ND), OP_LOGE(C5_INNER_DEBUG, "MmX Format should be ND."), return false);
        OP_TILING_CHECK(
            !(mmWeightFormat == ge::FORMAT_ND), OP_LOGE(C5_INNER_DEBUG, "MmWeight Format should be ND."), return false);
        OP_TILING_CHECK(
            !(mmYFormat == ge::FORMAT_ND), OP_LOGE(C5_INNER_DEBUG, "MmY Format should be ND."), return false);
    }
    return true;
}

static bool CheckDimNum(const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    auto gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    auto y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);

    OP_TILING_CHECK(
        gmmX->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE(C5_INNER_DEBUG, "GmmX's Dim should be 2, but get %lu.", gmmX->GetStorageShape().GetDimNum()),
        return false);
    OP_TILING_CHECK(
        gmmWeight->GetStorageShape().GetDimNum() != DIM_THREE,
        OP_LOGE(C5_INNER_DEBUG, "GmmWeight's Dim should be 3, but get %lu.", gmmWeight->GetStorageShape().GetDimNum()),
        return false);
    OP_TILING_CHECK(
        y->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE(C5_INNER_DEBUG, "Y's Dim should be 2, but get %lu.", y->GetStorageShape().GetDimNum()), return false);
    if (tilingData.commonTilingInfo.get_isOptionaSendRecvCountTensors()) {
        auto sendCounts = context->GetOptionalInputShape(SEND_COUNTS_TENSOR_INDEX);
        auto recvCounts = context->GetOptionalInputShape(RECV_COUNTS_TENSOR_INDEX);
        OP_TILING_CHECK(
            sendCounts->GetStorageShape().GetDimNum() != DIM_ONE,
            OP_LOGE(
                C5_INNER_DEBUG, "SendCountsTensor's Dim should be 1, but get %lu.",
                sendCounts->GetStorageShape().GetDimNum()),
            return false);
        OP_TILING_CHECK(
            recvCounts->GetStorageShape().GetDimNum() != DIM_ONE,
            OP_LOGE(
                C5_INNER_DEBUG, "RecvCountsTensor's Dim should be 1, but get %lu.",
                sendCounts->GetStorageShape().GetDimNum()),
            return false);
    }

    if (tilingData.commonTilingInfo.get_isOptionalMatmul()) {
        auto mmX = context->GetOptionalInputShape(MM_X_INDEX);
        auto mmWeight = context->GetOptionalInputShape(MM_WEIGHT_INDEX);
        auto mmY = context->GetOutputShape(OUTPUT_MM_Y_INDEX);
        OP_TILING_CHECK(
            mmX->GetStorageShape().GetDimNum() != DIM_TWO,
            OP_LOGE(C5_INNER_DEBUG, "MmX's Dim should be 2, but get %lu.", mmX->GetStorageShape().GetDimNum()),
            return false);
        OP_TILING_CHECK(
            mmWeight->GetStorageShape().GetDimNum() != DIM_TWO,
            OP_LOGE(
                C5_INNER_DEBUG, "MmWeight's Dim should be 2, but get %lu.", mmWeight->GetStorageShape().GetDimNum()),
            return false);
        OP_TILING_CHECK(
            mmY->GetStorageShape().GetDimNum() != DIM_TWO,
            OP_LOGE(C5_INNER_DEBUG, "MmY's Dim should be 2, but get %lu.", mmY->GetStorageShape().GetDimNum()),
            return false);
    }

    return true;
}

static bool CheckAndSetSendRecvCounts(
    const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    auto gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    auto y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);
    int64_t a = gmmX->GetStorageShape().GetDim(0);
    int64_t bsK = y->GetStorageShape().GetDim(0);
    int64_t eOverEp = gmmWeight->GetStorageShape().GetDim(0);

    auto attrs = context->GetAttrs();
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    size_t sendSize = sendCountsPtr->GetSize();
    OP_TILING_CHECK(
        sendSize != eOverEp * tilingData.commonTilingInfo.get_epWorldSize(),
        OP_LOGD(
            C5_INNER_DEBUG, "sendCounts size[%ld] doesnot equal to expert num[%ld]", sendSize,
            eOverEp * tilingData.commonTilingInfo.get_epWorldSize()),
        return false);
    const int64_t* sendArray = static_cast<const int64_t*>(sendCountsPtr->GetData());
    int64_t sendCountSum = 0;
    for (size_t i = 0; i < sendSize; i++) {
        sendCountSum += sendArray[i];
    }
    OP_TILING_CHECK(
        sendCountSum != a, OP_LOGE(C5_INNER_DEBUG, "a[%ld] should be equal to sum of sendCounts[%ld]", a, sendCountSum),
        return false);

    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    size_t recvSize = recvCountsPtr->GetSize();
    OP_TILING_CHECK(
        recvSize != eOverEp * tilingData.commonTilingInfo.get_epWorldSize(),
        OP_LOGD(
            C5_INNER_DEBUG, "recvCounts size[%ld] doesnot equal to expert num[%ld]", recvSize,
            eOverEp * tilingData.commonTilingInfo.get_epWorldSize()),
        return false);
    const int64_t* recvArray = static_cast<const int64_t*>(recvCountsPtr->GetData());
    int64_t recvCountSum = 0;
    for (size_t i = 0; i < recvSize; i++) {
        recvCountSum += recvArray[i];
    }
    OP_TILING_CHECK(
        recvCountSum != bsK,
        OP_LOGE(C5_INNER_DEBUG, "bsK[%ld] should be equal to sum of recvCounts[%ld]", bsK, recvCountSum), return false);

    std::copy_n(
        static_cast<const int64_t*>(recvCountsPtr->GetData()), recvCountsPtr->GetSize(),
        tilingData.aicpuTilingInfo.get_recvCnt());
    std::copy_n(
        static_cast<const int64_t*>(sendCountsPtr->GetData()), sendCountsPtr->GetSize(),
        tilingData.aicpuTilingInfo.get_sendCnt());
    return true;
}

static bool CheckDimValueIsNeedMM(const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);
    auto mmX = context->GetOptionalInputShape(MM_X_INDEX);
    auto mmWeight = context->GetOptionalInputShape(MM_WEIGHT_INDEX);
    auto mmY = context->GetOutputShape(OUTPUT_MM_Y_INDEX);

    int64_t bs = mmX->GetStorageShape().GetDim(0);
    int64_t sharedH = mmX->GetStorageShape().GetDim(1);
    int64_t bsK = y->GetStorageShape().GetDim(0);
    int64_t mmWeightH = tilingData.commonTilingInfo.get_isMmWeightTrans() ? mmWeight->GetStorageShape().GetDim(1) :
                                                                            mmWeight->GetStorageShape().GetDim(0);
    int64_t n2 = tilingData.commonTilingInfo.get_isMmWeightTrans() ? mmWeight->GetStorageShape().GetDim(0) :
                                                                     mmWeight->GetStorageShape().GetDim(1);
    int64_t mmYDim0 = mmY->GetStorageShape().GetDim(0);
    int64_t mmYDim1 = mmY->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(
        ((bs <= NUM_ZERO) || (bs >= MAX_BSK_VALUE)), OP_LOGE(C5_INNER_DEBUG, "bs[%ld] should be in (0, 52428800)!", bs),
        return false);

    int64_t k = bsK / bs;

    OP_TILING_CHECK(
        ((sharedH <= NUM_ZERO) || (sharedH > MAX_SHARED_H_SHAPE_SIZE)),
        OP_LOGE(C5_INNER_DEBUG, "H2[%ld] should be in (0, 12288]!", sharedH), return false);
    OP_TILING_CHECK(
        ((n2 <= NUM_ZERO) || (n2 >= MAX_DIM_VALUE)), OP_LOGE(C5_INNER_DEBUG, "N2[%ld] should be in (0, 65536)!", n2),
        return false);
    OP_TILING_CHECK(
        ((k < NUM_TWO) || (k > NUM_EIGHT)), OP_LOGE(C5_INNER_DEBUG, "K[%ld] should be in [2, 8]!", k), return false);

    OP_TILING_CHECK(
        sharedH != mmWeightH,
        OP_LOGE(C5_INNER_DEBUG, "mmWeightDim0[%ld] should equal to sharedH[%ld].", mmWeightH, sharedH), return false);
    OP_TILING_CHECK(
        n2 != mmYDim1, OP_LOGE(C5_INNER_DEBUG, "mmYDim1[%ld] should equal to N2[%ld].", mmYDim1, n2), return false);
    OP_TILING_CHECK(
        mmYDim0 != bs, OP_LOGE(C5_INNER_DEBUG, "mmYDim0[%ld] should equal to bs[%ld].", mmYDim0, bs), return false);
    OP_TILING_CHECK(
        sharedH > MAX_MATMUL_K,
        OP_LOGE(C5_INNER_DEBUG, "sharedH[%ld] exceeds matmul's limit[%ld]", sharedH, MAX_MATMUL_K), return false);
    tilingData.commonTilingInfo.set_Bs(bs);
    tilingData.commonTilingInfo.set_sharedMatmulH(sharedH);
    tilingData.commonTilingInfo.set_N2(n2);
    return true;
}

static bool CheckCntSum(
    const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto attrs = context->GetAttrs();
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    auto gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    auto y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);
    int64_t a = gmmX->GetStorageShape().GetDim(0);
    int64_t bsK = y->GetStorageShape().GetDim(0);
    int64_t eOverEp = gmmWeight->GetStorageShape().GetDim(0);
    int64_t epWorldSize = static_cast<int64_t>(tilingData.commonTilingInfo.get_epWorldSize());

    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    const int64_t* recvArray = static_cast<const int64_t*>(recvCountsPtr->GetData());
    const int64_t* sendArray = static_cast<const int64_t*>(sendCountsPtr->GetData());

    for (int64_t i = 1; i <= epWorldSize; i++) {
        for (int64_t j = (i - 1) * eOverEp; j <= i * eOverEp - 1; j++) {
            OP_TILING_CHECK(
                (sendArray[j] < NUM_ZERO) || (sendArray[j] > a),
                OP_LOGE(C5_INNER_DEBUG, "sendCounts[%ld] should be in [0, a[%ld]], but get %ld",j, a, sendArray[j]),
                return false);
            OP_TILING_CHECK(
                (recvArray[j] < NUM_ZERO) || (recvArray[j] > bsK),
                OP_LOGE(C5_INNER_DEBUG, "recvCounts[%ld] should be in [0, bsK[%ld]], but get %ld",j, bsK, recvArray[j]),
                return false);
        }
    }
    return true;
}

static bool CheckSendCntAndRecvCnt(
    const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto attrs = context->GetAttrs();
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    auto gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    auto y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);

    int64_t a = gmmX->GetStorageShape().GetDim(0);
    int64_t eOverEp = gmmWeight->GetStorageShape().GetDim(0);
    int64_t bsK = y->GetStorageShape().GetDim(0);
    int64_t epWorldSize = static_cast<int64_t>(tilingData.commonTilingInfo.get_epWorldSize());

    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    size_t recvSize = recvCountsPtr->GetSize();
    const int64_t* recvArray = static_cast<const int64_t*>(recvCountsPtr->GetData());
    size_t sendSize = sendCountsPtr->GetSize();
    const int64_t* sendArray = static_cast<const int64_t*>(sendCountsPtr->GetData());
    OP_TILING_CHECK(
        static_cast<int64_t>(recvSize) != epWorldSize * eOverEp,
        OP_LOGE(
            C5_INNER_DEBUG, "The length of recvCnts[%lu] should be equal to eOverEp * epworldSize[%ld]", recvSize,
            epWorldSize * eOverEp),
        return false);
    OP_TILING_CHECK(
        static_cast<int64_t>(sendSize) != epWorldSize * eOverEp,
        OP_LOGE(
            C5_INNER_DEBUG, "The length of sendCnts[%lu] should be equal to eOverEp * epworldSize[%ld]", sendSize,
            epWorldSize * eOverEp),
        return false);

    int64_t recvSum = 0;
    for (uint64_t i = 0; i < recvSize; i++) {
        recvSum += recvArray[i];
    }
    OP_TILING_CHECK(
        bsK != recvSum,
        OP_LOGE(C5_INNER_DEBUG, "bsK[%ld] should be equal to the sum of recvCounts[%ld]!", bsK, recvSum),
        return false);

    int64_t sendSum = 0;
    for (uint64_t i = 0; i < sendSize; i++) {
        sendSum += sendArray[i];
    }
    OP_TILING_CHECK(
        a != sendSum, OP_LOGE(C5_INNER_DEBUG, "a[%ld] should be equal to the sum of sendCounts[%ld]!", a, sendSum),
        return false);
    OP_TILING_CHECK(
        !CheckCntSum(context, tilingData),
        OP_LOGE(C5_INNER_DEBUG, "CheckCntSum failed!"), return false);
    return true;
}

static bool CheckDimValueShape(const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    auto gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    auto y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);

    int64_t h = gmmX->GetStorageShape().GetDim(1);
    int64_t eOverEp = gmmWeight->GetStorageShape().GetDim(0);
    int64_t gmmWeightH = tilingData.commonTilingInfo.get_isGmmWeightTrans() ? gmmWeight->GetStorageShape().GetDim(2) :
                                                                              gmmWeight->GetStorageShape().GetDim(1);
    int64_t n1 = tilingData.commonTilingInfo.get_isGmmWeightTrans() ? gmmWeight->GetStorageShape().GetDim(1) :
                                                                      gmmWeight->GetStorageShape().GetDim(2);
    int64_t bsK = y->GetStorageShape().GetDim(0);
    int64_t yDim1 = y->GetStorageShape().GetDim(1);
    int64_t epWorldSize = static_cast<int64_t>(tilingData.commonTilingInfo.get_epWorldSize());
    OP_TILING_CHECK(
        h != gmmWeightH, OP_LOGE(C5_INNER_DEBUG, "gmmWeightDim1[%ld] should equal to h[%ld].", gmmWeightH, h),
        return false);
    OP_TILING_CHECK(
        n1 != yDim1, OP_LOGE(C5_INNER_DEBUG, "n1[%ld] should equal to yDim1[%ld].", n1, yDim1), return false);
    OP_TILING_CHECK(
        h > MAX_MATMUL_K, OP_LOGE(C5_INNER_DEBUG, "h[%ld] exceeds matmul's limit[%ld]", h, MAX_MATMUL_K), return false);
    OP_TILING_CHECK(eOverEp > MAX_E,
        OP_LOGE(C5_INNER_DEBUG, "eOverEp[%ld] exceeds limit[%ld]", eOverEp, MAX_E), return false);
    OP_TILING_CHECK(
        ((bsK <= NUM_ZERO) || (bsK >= MAX_BSK_VALUE)),
        OP_LOGE(C5_INNER_DEBUG, "bsK[%ld] should be in (0, 52428800)!", bsK),
        return false);
    OP_TILING_CHECK(
        ((h <= NUM_ZERO) || (h >= MAX_DIM_VALUE)), OP_LOGE(C5_INNER_DEBUG, "H1[%ld] should be in (0, 65536)!", h),
        return false);
    OP_TILING_CHECK(
        ((gmmWeightH <= NUM_ZERO) || (gmmWeightH > MAX_SHARED_H_SHAPE_SIZE)),
        OP_LOGE(C5_INNER_DEBUG, "H2[%ld] should be in (0, 12288]!", h),
        return false);
    OP_TILING_CHECK(
        ((n1 <= NUM_ZERO) || (n1 >= MAX_DIM_VALUE)), OP_LOGE(C5_INNER_DEBUG, "n1[%ld] should be in (0, 65536)!", n1),
        return false);
    OP_TILING_CHECK(
        ((eOverEp <= NUM_ZERO) || (eOverEp > MAX_EXPERT_NUM_PER_RANK)),
        OP_LOGE(C5_INNER_DEBUG, "eOverEp[%ld] should be in (0, 32]!", eOverEp), return false);

    OP_TILING_CHECK(
        !CheckSendCntAndRecvCnt(context, tilingData),
        OP_LOGE(C5_INNER_DEBUG, "CheckSendCntAndRecvCnt failed!"), return false);
    
    std::vector<int64_t> epWorldSizeOptional{2, 4, 8, 16, 32, 64};
    OP_TILING_CHECK(
        std::find(epWorldSizeOptional.begin(), epWorldSizeOptional.end(), epWorldSize) == epWorldSizeOptional.end(),
        OP_LOGE(C5_INNER_DEBUG, "epWorldSize[%ld] should be 2,4,8,16,32,64!", epWorldSize), return false);
    return true;
}

static bool CheckDimValue(const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto gmmX = context->GetInputShape(GMM_X_INDEX);
    auto gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    auto y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);

    int64_t a = gmmX->GetStorageShape().GetDim(0);
    int64_t eOverEp = gmmWeight->GetStorageShape().GetDim(0);
    int64_t n1 = tilingData.commonTilingInfo.get_isGmmWeightTrans() ? gmmWeight->GetStorageShape().GetDim(1) :
                                                                      gmmWeight->GetStorageShape().GetDim(2);
    int64_t h = gmmX->GetStorageShape().GetDim(1);
    int64_t bsK = y->GetStorageShape().GetDim(0);

    OP_TILING_CHECK(
        !CheckDimValueShape(context, tilingData), OP_LOGE(C5_INNER_DEBUG, "CheckDimValueShape failed."),
        return false);

    OP_TILING_CHECK(
        !CheckAndSetSendRecvCounts(context, tilingData), OP_LOGE(C5_INNER_DEBUG, "CheckSendRecvCounts failed."),
        return false);

    tilingData.commonTilingInfo.set_BsK(bsK);
    tilingData.commonTilingInfo.set_H(h);
    tilingData.commonTilingInfo.set_E_ep(eOverEp);
    tilingData.commonTilingInfo.set_A(a);
    tilingData.commonTilingInfo.set_N1(n1);

    if (tilingData.commonTilingInfo.get_isOptionalMatmul()) {
        OP_TILING_CHECK(
            !CheckDimValueIsNeedMM(context, tilingData), OP_LOGE(C5_INNER_DEBUG, "CheckDimValueIsNeedMM  failed"),
            return false);
    }
    return true;
}

static bool CheckAndSetAttrs(const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(C5_INNER_DEBUG, "attrs is nullptr!"), return false);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_EP_WORLD_SIZE_INDEX);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);

    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE(C5_INNER_DEBUG, "groupEpPtr is nullptr!"), return false);
    OP_TILING_CHECK(
        epWorldSizePtr == nullptr, OP_LOGE(C5_INNER_DEBUG, "epWorldSizePtr is nullptr!"), return false);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE(C5_INNER_DEBUG, "sendCountsPtr is nullptr!"), return false);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE(C5_INNER_DEBUG, "recvCountsPtr is nullptr!"), return false);

    tilingData.commonTilingInfo.set_epWorldSize(*epWorldSizePtr);
    tilingData.commonTilingInfo.set_isGmmWeightTrans(*transGmmWeightPtr);
    tilingData.commonTilingInfo.set_isMmWeightTrans(*transMmWeightPtr);

    return true;
}

static bool CheckInputAndOutput(const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    const gert::StorageShape* gmmX = context->GetInputShape(GMM_X_INDEX);
    const gert::StorageShape* gmmWeight = context->GetInputShape(GMM_WEIGHT_INDEX);
    const gert::StorageShape* sendCounts = context->GetOptionalInputShape(SEND_COUNTS_TENSOR_INDEX);
    const gert::StorageShape* recvCounts = context->GetOptionalInputShape(RECV_COUNTS_TENSOR_INDEX);
    const gert::StorageShape* mmX = context->GetOptionalInputShape(MM_X_INDEX);
    const gert::StorageShape* mmWeight = context->GetOptionalInputShape(MM_WEIGHT_INDEX);
    const gert::StorageShape* y = context->GetOutputShape(OUTPUT_GMM_Y_INDEX);
    const gert::StorageShape* mmY = context->GetOutputShape(OUTPUT_MM_Y_INDEX);

    OP_TILING_CHECK(gmmX == nullptr, OP_LOGE(C5_INNER_DEBUG, "GmmX is nullptr!"), return false);
    OP_TILING_CHECK(gmmWeight == nullptr, OP_LOGE(C5_INNER_DEBUG, "GmmWeight is nullptr!"), return false);
    OP_TILING_CHECK(y == nullptr, OP_LOGE(C5_INNER_DEBUG, "GmmY is nullptr!"), return false);

    if ((sendCounts != nullptr) || (recvCounts != nullptr)) {
        OP_LOGW(C5_INNER_DEBUG, "sendCountsTensorOptional and recvCountsTensorOptional should be nullptr.");
    }
    tilingData.commonTilingInfo.set_isOptionaSendRecvCountTensors(false);

    if (mmX != nullptr) {
        OP_TILING_CHECK(
            mmWeight == nullptr, OP_LOGE(C5_INNER_DEBUG, "mmWeight is nullptr when mmX is given!"), return false);
        OP_TILING_CHECK(mmY == nullptr, OP_LOGE(C5_INNER_DEBUG, "mmY is nullptr when mmX is given!"), return false);
    }
    tilingData.commonTilingInfo.set_isOptionalMatmul(context->GetOptionalInputShape(MM_X_INDEX) != nullptr);

    auto gmmXInputDesc = context->GetInputDesc(GMM_X_INDEX);
    OP_TILING_CHECK(gmmXInputDesc == nullptr, OP_LOGE(C5_INNER_DEBUG, "gmmXInputDesc is nullptr!"), return false);
    auto gmmWeightInputDesc = context->GetInputDesc(GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmWeightInputDesc == nullptr, 
        OP_LOGE(C5_INNER_DEBUG, "gmmWeightInputDesc is nullptr!"), return false);
    auto gmmYOutputDesc = context->GetOutputDesc(OUTPUT_GMM_Y_INDEX);
    OP_TILING_CHECK(gmmYOutputDesc == nullptr, OP_LOGE(C5_INNER_DEBUG, "gmmYOutputDesc is nullptr!"), return false);

    if (tilingData.commonTilingInfo.get_isOptionaSendRecvCountTensors()) {
        auto recvCntsOptionalInputDesc = context->GetOptionalInputDesc(RECV_COUNTS_TENSOR_INDEX);
        OP_TILING_CHECK(recvCntsOptionalInputDesc == nullptr, 
            OP_LOGE(C5_INNER_DEBUG, "recvCntsOptionalInputDesc is nullptr!"), return false);
        auto sendCntsOptionalInputDesc = context->GetOptionalInputDesc(SEND_COUNTS_TENSOR_INDEX);
        OP_TILING_CHECK(sendCntsOptionalInputDesc == nullptr, 
            OP_LOGE(C5_INNER_DEBUG, "sendCntsOptionalInputDesc is nullptr!"), return false);
    }

    if (tilingData.commonTilingInfo.get_isOptionalMatmul()) {
        auto mmXOptionalInputDesc = context->GetOptionalInputDesc(MM_X_INDEX);
        OP_TILING_CHECK(mmXOptionalInputDesc == nullptr, 
            OP_LOGE(C5_INNER_DEBUG, "mmXOptionalInputDesc is nullptr!"), return false);
        auto mmWeightOptionalInputDesc = context->GetOptionalInputDesc(MM_WEIGHT_INDEX);
        OP_TILING_CHECK(mmWeightOptionalInputDesc == nullptr, 
            OP_LOGE(C5_INNER_DEBUG, "mmWeightOptionalInputDesc is nullptr!"), return false);
        auto mmYOptionalInputDesc = context->GetOutputDesc(OUTPUT_MM_Y_INDEX);
        OP_TILING_CHECK(mmYOptionalInputDesc == nullptr, 
            OP_LOGE(C5_INNER_DEBUG, "mmYOptionalInputDesc is nullptr!"), return false);
    }

    OP_TILING_CHECK(
        !CheckDType(context, tilingData), OP_LOGE(C5_INNER_DEBUG, "CheckDTypeAndFormat failed!"), return false);
    OP_TILING_CHECK(
        !CheckFormat(context, tilingData), OP_LOGE(C5_INNER_DEBUG, "CheckDTypeAndFormat failed!"), return false);
    OP_TILING_CHECK(!CheckDimNum(context, tilingData), OP_LOGE(C5_INNER_DEBUG, "CheckDimNum failed!"), return false);
    OP_TILING_CHECK(!CheckDimValue(context, tilingData), OP_LOGE(C5_INNER_DEBUG, "CheckDimNum failed!"), return false);
    return true;
}

static void SetHcclTiling(const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    tilingData.set_version(HCCL_VERSION);
    tilingData.set_hcommCnt(1);

    auto gmmX = context->GetInputDesc(GMM_X_INDEX);
    OP_TILING_CHECK(gmmX == nullptr, OP_LOGE(C5_INNER_DEBUG, "gmmX is nullptr!"), return );

    tilingData.hcommCfgATA.set_srcDataType(static_cast<uint32_t>(
        mc2tiling::ConvertGeTypeToHcclType(C5_INNER_DEBUG, gmmX->GetDataType())));
    tilingData.hcommCfgATA.set_dstDataType(static_cast<uint32_t>(
        mc2tiling::ConvertGeTypeToHcclType(C5_INNER_DEBUG, gmmX->GetDataType())));
    tilingData.hcommCfgATA.set_opType(HCCL_CMD_ALLTOALLV);
}

static ge::graphStatus ComputeBaseMNK(
    const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);

    maxM = tilingData.commonTilingInfo.get_A();
    maxN = tilingData.commonTilingInfo.get_N1();
    maxK = tilingData.commonTilingInfo.get_H();
    uint32_t baseN = BEST_BASE_N;
    while (baseN > static_cast<uint32_t>(maxN)) {
        baseN = baseN >> 1;
    }
    if (baseN < static_cast<uint32_t>(maxN)) {
        baseN = baseN << 1;
    }
    baseN_ = std::min<int32_t>(BEST_BASE_N, baseN);

    baseK_ = (PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B) / (baseN_ * FP16_DATASIZE);
    baseK_ = SixteenAlign(baseK_);
    if (baseK_ > MAX_BASE_K) {
        baseK_ = MAX_BASE_K;
        int32_t maxBaseN = SixteenAlign(PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B / (baseK_ * FP16_DATASIZE));
        baseN_ = std::min<int32_t>(baseN_, maxBaseN);
        baseN_ = std::max<int32_t>(16, static_cast<int32_t>(SixteenAlign(baseN_, true))); // 16: min value for baseN
    }
    if (baseK_ > maxK) {
        baseK_ = std::min<int32_t>(baseK_, SixteenAlign(maxK, true));
    }
    OP_TILING_CHECK(baseK_ == 0, OP_LOGE(C5_INNER_DEBUG, "baseK cannot be 0."), return ge::GRAPH_FAILED);

    // 基于使能 double buffer的L0A内存与L0B内存计算BaseM(cube)
    uint32_t maxBaseM = PLATFORM_SIZE.l0CSize / (baseN_ * FP16_DATASIZE);
    baseM_ = std::min<uint32_t>((PLATFORM_SIZE.l0ASize / DOUBLE_BUFFER_L0A_L0B / (baseK_ * FP16_DATASIZE)), maxBaseM);
    baseM_ = SixteenAlign(baseM_);
    if (baseM_ > maxM) {
        baseM_ = SixteenAlign(maxM, true);
    }
    OP_TILING_CHECK(baseM_ == 0, OP_LOGE(C5_INNER_DEBUG, "curBaseM cannot be 0."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeSharedBaseMNK(
    const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);

    maxM = tilingData.commonTilingInfo.get_Bs();
    maxN = tilingData.commonTilingInfo.get_N2();
    maxK = tilingData.commonTilingInfo.get_sharedMatmulH();
    uint32_t baseN = BEST_BASE_N;
    while (baseN > static_cast<uint32_t>(maxN)) {
        baseN = baseN >> 1;
    }
    if (baseN < static_cast<uint32_t>(maxN)) {
        baseN = baseN << 1;
    }
    baseN_ = std::min<int32_t>(BEST_BASE_N, baseN);

    baseK_ = (PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B) / (baseN_ * FP16_DATASIZE);
    baseK_ = SixteenAlign(baseK_);
    if (baseK_ > MAX_BASE_K) {
        baseK_ = MAX_BASE_K;
        int32_t maxBaseN = SixteenAlign(PLATFORM_SIZE.l0BSize / DOUBLE_BUFFER_L0A_L0B / (baseK_ * FP16_DATASIZE));
        baseN_ = std::min<int32_t>(baseN_, maxBaseN);
        baseN_ = std::max<int32_t>(16, SixteenAlign(baseN_, true)); // 16: min value for baseN
    }
    if (baseK_ > maxK) {
        baseK_ = std::min<int32_t>(baseK_, SixteenAlign(maxK, true));
    }
    OP_TILING_CHECK(baseK_ == 0, OP_LOGE(C5_INNER_DEBUG, "baseK cannot be 0."), return ge::GRAPH_FAILED);

    // 基于使能 double buffer的L0A内存与L0B内存计算BaseM(cube)
    uint32_t maxBaseM = PLATFORM_SIZE.l0CSize / (baseN_ * FP16_DATASIZE);
    baseM_ = std::min<uint32_t>((PLATFORM_SIZE.l0ASize / DOUBLE_BUFFER_L0A_L0B / (baseK_ * FP16_DATASIZE)), maxBaseM);
    baseM_ = SixteenAlign(baseM_);
    if (baseM_ > maxM) {
        baseM_ = SixteenAlign(maxM, true);
    }
    OP_TILING_CHECK(baseM_ == 0, OP_LOGE(C5_INNER_DEBUG, "baseM cannot be 0."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoMatmulApiTiling(
    const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto gmmX = context->GetInputDesc(GMM_X_INDEX);
    OP_TILING_CHECK(gmmX == nullptr, OP_LOGE(C5_INNER_DEBUG, "gmmX is nullptr!"), return ge::GRAPH_FAILED);

    auto matmulDType = (gmmX->GetDataType() == ge::DT_FLOAT16) ?
                           matmul_tiling::DataType::DT_FLOAT16 :
                           matmul_tiling::DataType::DT_BFLOAT16;

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDType, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDType, false);
    mm.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, matmulDType);
    mm.SetOrgShape(maxM, maxN, maxK);
    mm.SetShape(maxM, baseN_, maxK);
    mm.SetFixSplit(std::min(baseM_, maxM), baseN_);
    mm.SetBufferSpace(PLATFORM_SIZE.l1Size, PLATFORM_SIZE.l0CSize, ubSize);

    OP_TILING_CHECK(
        mm.GetTiling(tilingData.matmulTiling) == -1, OP_LOGE(C5_INNER_DEBUG, "matmul api getTiling failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoSharedMatmulApiTiling(
    const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    auto mmX = context->GetInputDesc(MM_X_INDEX);
    OP_TILING_CHECK(mmX == nullptr, OP_LOGE(C5_INNER_DEBUG, "mmX is nullptr!"), return ge::GRAPH_FAILED);

    auto matmulDType = (mmX->GetDataType() == ge::DT_FLOAT16) ?
                           matmul_tiling::DataType::DT_FLOAT16 :
                           matmul_tiling::DataType::DT_BFLOAT16;

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    matmul_tiling::MatmulApiTiling mm(ascendcPlatform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDType, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDType, false);
    mm.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, matmulDType);
    mm.SetOrgShape(maxM, maxN, maxK);
    mm.SetShape(maxM, baseN_, maxK);
    mm.SetFixSplit(std::min(baseM_, maxM), baseN_);
    mm.SetBufferSpace(PLATFORM_SIZE.l1Size, PLATFORM_SIZE.l0CSize, ubSize);

    OP_TILING_CHECK(
        mm.GetTiling(tilingData.sharedExpMatmulTiling) == -1,
        OP_LOGE(C5_INNER_DEBUG, "sharedExpMatmul api getTiling failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetMatmulTiling(
    const gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    OP_TILING_CHECK(
        ComputeBaseMNK(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(C5_INNER_DEBUG, "GMM tiling compute baseMNK failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        DoMatmulApiTiling(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(C5_INNER_DEBUG, "GMM tiling compute baseMNK failed."), return ge::GRAPH_FAILED);

    if (tilingData.commonTilingInfo.get_isOptionalMatmul()) {
        OP_TILING_CHECK(
            ComputeSharedBaseMNK(context, tilingData) != ge::GRAPH_SUCCESS,
            OP_LOGE(C5_INNER_DEBUG, "GMM tiling compute baseMNK failed."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            DoSharedMatmulApiTiling(context, tilingData) != ge::GRAPH_SUCCESS,
            OP_LOGE(C5_INNER_DEBUG, "GMM tiling compute baseMNK failed."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(
        workspaceSize == nullptr, OP_LOGE(C5_INNER_DEBUG, "workspace is nullptr."), return ge::GRAPH_FAILED);
    uint64_t gmmOut = tilingData.commonTilingInfo.get_A() * tilingData.commonTilingInfo.get_N1() * FP16_DATASIZE;
    workspaceSize[0] = SYS_WORKSPACE_SIZE + gmmOut;
    OP_LOGD(C5_INNER_DEBUG, "gmmOut %lu, workspace %lu", gmmOut, static_cast<uint64_t>(workspaceSize[0]));
    return ge::GRAPH_SUCCESS;
}

static void UpdateTilingKey(uint64_t& tilingKey, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    tilingKey += (tilingData.commonTilingInfo.get_isOptionalMatmul()) ? TILINGKEY_COMPUTE_OPTIONAL_MM : 0UL;
    tilingKey += (tilingData.commonTilingInfo.get_isGmmWeightTrans()) ? TILINGKEY_GMM_WEIGHT_TRANSPOSE : 0UL;
    if (tilingData.commonTilingInfo.get_isOptionalMatmul()) {
        tilingKey += (tilingData.commonTilingInfo.get_isMmWeightTrans()) ? TILINGKEY_MM_WEIGHT_TRANSPOSE : 0UL;
    }
    tilingKey += TILINGKEY_A5_OFFSET;
}

static void SetTilingData(gert::TilingContext* context, GroupedMatMulAlltoAllvTilingDataA5& tilingData)
{
    OP_TILING_CHECK(context->GetRawTilingData() == nullptr, 
                    OP_LOGE(C5_INNER_DEBUG, "RawTilingData of GroupedMatMulAlltoAllv is nullptr!"), return );
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus GroupedMatMulAlltoAllvTilingFuncA5(gert::TilingContext* context)
{
    GroupedMatMulAlltoAllvTilingDataA5 tilingData;

    OP_TILING_CHECK(
        !CheckAndSetAttrs(context, tilingData), OP_LOGE(C5_INNER_DEBUG, "CheckAndSetAttrs failed."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !CheckInputAndOutput(context, tilingData), OP_LOGE(C5_INNER_DEBUG, "CheckInputAndOutput failed."),
        return ge::GRAPH_FAILED);

    // 设置 CV 核数
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);
    uint64_t ubSize = 0UL;
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);
    tilingData.commonTilingInfo.set_aivCoreNum(aivNum);
    tilingData.commonTilingInfo.set_aicCoreNum(aicNum);

    SetHcclTiling(context, tilingData);

    OP_TILING_CHECK(
        SetMatmulTiling(context, tilingData) != ge::GRAPH_SUCCESS, OP_LOGE(C5_INNER_DEBUG, "SetMatmulTiling failed."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        SetWorkspace(context, tilingData) != ge::GRAPH_SUCCESS, OP_LOGE(C5_INNER_DEBUG, "SetWorkspace failed."),
        return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0U;

    if (!tilingData.commonTilingInfo.get_isOptionalMatmul()) {
            OP_TILING_CHECK(
            (context->GetOutputShape(OUTPUT_MM_Y_INDEX) != nullptr &&
             context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum() != NUM_ZERO),
            OP_LOGE(C5_INNER_DEBUG, "The mmY should be null when mmX and mmWeight are null!"), return ge::GRAPH_FAILED);
        if (tilingData.commonTilingInfo.get_isMmWeightTrans()) {
            OP_LOGE(C5_INNER_DEBUG, "The trans_mm_weight should be false when mmX mmWeight mmY is null!");
            return ge::GRAPH_FAILED;
        }
    }

    UpdateTilingKey(tilingKey, tilingData);
    OP_LOGD(C5_INNER_DEBUG, "Compute tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    PrintTilingData(tilingData);

    SetTilingData(context, tilingData);
    OP_LOGD("GroupedMatMulAlltoAllv", "tiling process finshed successfully.");
    return ge::GRAPH_SUCCESS;
}

bool GmmAlltoAllvTilingA5::IsCapable()
{
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND910_95) {
        OP_LOGD(C5_INNER_DEBUG, "Do GmmAlltoAllvTilingA5 tiling.");
        return true;
    }
    return false;
}

ge::graphStatus GmmAlltoAllvTilingA5::DoOpTiling()
{
    return GroupedMatMulAlltoAllvTilingFuncA5(context_);
}

uint64_t GmmAlltoAllvTilingA5::GetTilingKey() const
{
    // tilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(C5_INNER_DEBUG, "GmmAlltoAllvTilingA5 get tiling key %lu", tilingKey);
    return tilingKey;
}
} // namespace optiling