/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

/* !
 * \file fallback_allto_allv_grouped_mat_mul.cpp
 * \brief fallback function of op AlltoAllvGroupedMatMul
 */
#include "fallback/fallback_comm.h"
#include "fallback_opapi.h"
#include "op_mc2.h"
#include "error/ops_error.h"

namespace fallback
{
static ge::graphStatus AlltoAllvGroupedMatMulExecuteFunc(gert::OpExecuteContext* host_api_ctx)
{
    OP_LOGD("AlltoAllvGroupedMatMulFallback", "Start AlltoAllvGroupedMatMulFallback.");
    OP_CHECK(host_api_ctx == nullptr, OP_LOGE("AlltoAllvGroupedMatMulFallback", "host_api_ctx is null"),
             return ge::GRAPH_FAILED);

    const auto gmmX = host_api_ctx->GetInputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_GMM_X));
    OP_CHECK(gmmX == nullptr, OP_LOGE("AlltoAllvGroupedMatMulFallback", "gmmX is null"), return ge::GRAPH_FAILED);

    const auto gmmWeight =
        host_api_ctx->GetInputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_GMM_WEIGHT));
    OP_CHECK(gmmWeight == nullptr, OP_LOGE("AlltoAllvGroupedMatMulFallback", "gmmWeight is null"),
             return ge::GRAPH_FAILED);

    const auto sendCountsTensor = host_api_ctx->GetOptionalInputTensor(
        static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_SEND_COUNTS_TENSOR));

    const auto recvCountsTensor = host_api_ctx->GetOptionalInputTensor(
        static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_RECV_COUNTS_TENSOR));

    const auto mmX =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_MM_X));

    const auto mmWeight =
        host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulInputIdx::K_MM_WEIGHT));

    auto gmm_y = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulOutputIdx::K_GMM_Y));
    OP_CHECK(gmm_y == nullptr, OP_LOGE("AlltoAllvGroupedMatMulFallback", "gmm_y is null"), return ge::GRAPH_FAILED);

    auto mmY = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulOutputIdx::K_MM_Y));
    if (mmY != nullptr && mmY->GetStorageShape().GetDimNum() == 0) {
        mmY = nullptr;
    }

    auto permuteOut =
        host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::AlltoAllvGroupedMatMulOutputIdx::K_PERMUTE_OUT));
    if (permuteOut != nullptr && permuteOut->GetStorageShape().GetDimNum() == 0) {
        permuteOut = nullptr;
    }

    const auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK(attrs == nullptr, OP_LOGE("AlltoAllvGroupedMatMulFallback", "attrs is null"), return ge::GRAPH_FAILED);

    const auto group = attrs->GetStr(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_GROUP));
    OP_CHECK(group == nullptr, OP_LOGE("AlltoAllvGroupedMatMulFallback", "group is null"), return ge::GRAPH_FAILED);

    const auto epWorldSize = attrs->GetInt(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_EP_WORLD_SIZE));
    OP_CHECK(epWorldSize == nullptr, OP_LOGE("AlltoAllvGroupedMatMulFallback", "epWorldSize is null"),
             return ge::GRAPH_FAILED);

    const auto sendCounts = attrs->GetListInt(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_SEND_COUNTS));
    OP_CHECK(sendCounts == nullptr, OP_LOGE("AlltoAllvGroupedMatMulFallback", "sendCounts is null"),
             return ge::GRAPH_FAILED);
    std::vector<int64_t> actSendCountsSeqArray;
    const int64_t* actSendSeqData = sendCounts->GetData();
    const size_t sendLen = static_cast<size_t>(sendCounts->GetSize());
    for (size_t i = 0UL; i < sendLen; i++) {
        actSendCountsSeqArray.push_back(actSendSeqData[i]);
    }

    const auto recvCounts = attrs->GetListInt(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_RECV_COUNTS));
    OP_CHECK(recvCounts == nullptr, OP_LOGE("AlltoAllvGroupedMatMulFallback", "recvCounts is null"),
             return ge::GRAPH_FAILED);
    std::vector<int64_t> actRecvCountsSeqArray;
    const int64_t* actRecvSeqData = recvCounts->GetData();
    const size_t recvLen = static_cast<size_t>(recvCounts->GetSize());
    for (size_t i = 0UL; i < recvLen; i++) {
        actRecvCountsSeqArray.push_back(actRecvSeqData[i]);
    }

    const auto transGmmWeight =
        attrs->GetBool(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_TRANS_GMM_WEIGHT));
    OP_CHECK(transGmmWeight == nullptr, OP_LOGE("AlltoAllvGroupedMatMulFallback", "transGmmWeight is null"),
             return ge::GRAPH_FAILED);

    const auto transMmWeight =
        attrs->GetBool(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_TRANS_MM_WEIGHT));

    const auto permuteOutFlag =
        attrs->GetBool(static_cast<size_t>(ops::AlltoAllvGroupedMatMulAttrIdx::K_PERMUTE_OUT_FLAG));

    const auto api_ret =
        EXEC_OPAPI_CMD(aclnnAlltoAllvGroupedMatMul, gmmX, gmmWeight, sendCountsTensor, recvCountsTensor, mmX, mmWeight,
                       group, *epWorldSize, actSendCountsSeqArray, actRecvCountsSeqArray, *transGmmWeight,
                       *transMmWeight, *permuteOutFlag, gmm_y, mmY, permuteOut);
    OP_CHECK(api_ret != ge::GRAPH_SUCCESS,
             OP_LOGE("AlltoAllvGroupedMatMulFallback", "Aclnn api error code %d", api_ret), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AlltoAllvGroupedMatMul).OpExecuteFunc(AlltoAllvGroupedMatMulExecuteFunc);
}  // namespace fallback