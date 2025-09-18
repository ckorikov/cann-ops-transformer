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
 * \file moe_gating_top_k_softmax_v2_tiling_k_fullload.cpp
 * \brief
 */
// #include "graph/utils/op_desc_utils.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
// #include "external/exe_graph/runtime/shape.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling/tiling_type.h"
#include "tiling/tiling_api.h"
// #include "op_tiling_util.h"
#include "moe_gating_top_k_softmax_v2_tiling.h"
#include "log/log.h"
using namespace Ops::Transformer::OpTiling;
using namespace AscendC;
using namespace ge;

namespace optiling {
static const int32_t FP32_SIZE = 4;
static const int32_t MAX_COL_IN_UB = 8160; // ubSize/minTypeSize

class MoeGatingTopKSoftmaxV2KFullLoadTiling : public MoeGatingTopKSoftmaxV2BaseTiling
{
public:
    explicit MoeGatingTopKSoftmaxV2KFullLoadTiling(gert::TilingContext* context)
        : MoeGatingTopKSoftmaxV2BaseTiling(context)
    {}

protected:
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    uint32_t ubOuter;
    uint32_t ubLoop;
    uint32_t ubFormer;
    uint32_t ubFormerAlign;
    uint32_t ubTail;
    uint32_t ubTailAlign;
    MoeGatingTopKSoftmaxV2KFullLoadTilingData tilingData;
    bool CalculateParamInUb();
    uint64_t GetTotalTmpSizeInUb(uint32_t kAlign);
};

bool MoeGatingTopKSoftmaxV2KFullLoadTiling::IsCapable()
{
    return renorm == 0;
}

uint64_t MoeGatingTopKSoftmaxV2KFullLoadTiling::GetTotalTmpSizeInUb(uint32_t kAlign)
{
    int32_t dataTypeSize = FP32_SIZE;
    uint64_t gatingUbTmpSize = ubFormerAlign * dataTypeSize;
    uint64_t topkOutUbTmpSize = (ubFormerAlign + kAlign) * dataTypeSize;
    uint64_t topkIndicesOutUbTmpSize = (ubFormerAlign + kAlign) * FP32_SIZE;

    uint64_t finishUbTmpSize = ALIGN_NUM;
    auto shape = ge::Shape({ubFormerAlign});
    uint64_t softmaxUbTmpSize = GetSoftMaxFlashV2MaxTmpSize(shape, dataTypeSize, dataTypeSize, true);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    (void)GetTopKMaxMinTmpSize(
        ascendcPlatform, kAlign + ubFormerAlign, 1, false, true, TopKMode::TOPK_NORMAL, true, dataTypeSize, maxValue,
        minValue);
    maxValue = maxValue > softmaxUbTmpSize ? maxValue : softmaxUbTmpSize;

    uint64_t additionalUbtmpSize = ALIGN_NUM + ALIGN_NUM + ALIGN_NUM;

    uint64_t totalSize =
        gatingUbTmpSize + topkOutUbTmpSize + topkIndicesOutUbTmpSize + finishUbTmpSize + maxValue + additionalUbtmpSize;
    return totalSize;
}

bool MoeGatingTopKSoftmaxV2KFullLoadTiling::CalculateParamInUb()
{
    ubOuter = 1;
    int doubleBuffer = 2;
    if (col > MAX_COL_IN_UB) {
        ubOuter = col / MAX_COL_IN_UB;
    }
    uint32_t kAlign = CeilDiv(k, ALIGN_NUM) * ALIGN_NUM;
    while (true) {
        ubFormer = col / ubOuter;
        ubFormerAlign = CeilDiv(ubFormer, ALIGN_NUM) * ALIGN_NUM;
        uint64_t totalSize = GetTotalTmpSizeInUb(kAlign);
        if (totalSize <= (ubSize / doubleBuffer)) {
            break;
        }
        if (ubFormerAlign < kAlign) {
            ubOuter = 0;
            break;
        }
        ubOuter++;
    }
    if (ubOuter == 0) {
        return false;
    }
    auto ubFormerDownAlign = (ubFormer / ALIGN_NUM) * ALIGN_NUM;
    auto ubOuterDownAlign = (col + ubFormerDownAlign - 1) / ubFormerDownAlign;
    ubLoop = ubOuterDownAlign;
    ubFormer = ubFormerDownAlign;
    ubFormerAlign = ubFormer;
    ubTail = col - (ubLoop - 1) * ubFormerAlign;
    ubTailAlign = CeilDiv(ubTail, ALIGN_NUM) * ALIGN_NUM;
    return true;
}

ge::graphStatus MoeGatingTopKSoftmaxV2KFullLoadTiling::DoOpTiling()
{
    tilingData.set_row(row);
    tilingData.set_col(col);
    tilingData.set_k(k);
    tilingData.set_kAlign(CeilDiv(k, ALIGN_NUM) * ALIGN_NUM);
    tilingData.set_blockFormer(CeilDiv(row, coreNum));
    tilingData.set_blockNum(CeilDiv(row, tilingData.get_blockFormer()));
    tilingData.set_blockTail(row - (tilingData.get_blockNum() - 1) * tilingData.get_blockFormer());

    if (!CalculateParamInUb()) {
        OP_LOGE("[MoeGatingTopKSoftmaxV2K NonRenorm]", "AutoTiling failed, the K is too large.");
        return ge::GRAPH_FAILED;
    }
    tilingData.set_ubLoop(ubLoop);
    tilingData.set_ubFormer(ubFormer);
    tilingData.set_ubFormerAlign(ubFormerAlign);
    tilingData.set_ubTail(ubTail);
    tilingData.set_ubTailAlign(ubTailAlign);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2KFullLoadTiling::DoLibApiTiling()
{
    int dataTypeSize = FP32_SIZE;
    uint32_t ubFormerAlign = CeilDiv(ubFormer, ALIGN_NUM) * ALIGN_NUM;
    uint32_t kAlign = CeilDiv(k, ALIGN_NUM) * ALIGN_NUM;
    auto softmaxShape = ge::Shape({tilingData.get_ubFormer()});
    SoftMaxFlashV2TilingFunc(
        softmaxShape, dataTypeSize, dataTypeSize, GetSoftMaxFlashV2MaxTmpSize(softmaxShape, dataTypeSize, true, true),
        tilingData.ubFormerSoftmaxTilingData, true);

    softmaxShape = ge::Shape({tilingData.get_ubTail()});
    SoftMaxFlashV2TilingFunc(
        softmaxShape, dataTypeSize, dataTypeSize, GetSoftMaxFlashV2MaxTmpSize(softmaxShape, dataTypeSize, true, true),
        tilingData.ubTailSoftmaxTilingData, true);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

    TopKTilingFunc(
        ascendcPlatform, kAlign + ubFormerAlign, 1, kAlign, dataTypeSize, true, TopKMode::TOPK_NORMAL, true,
        tilingData.topkFormerTilingData);

    TopKTilingFunc(
        ascendcPlatform, kAlign + ubTailAlign, 1, kAlign, dataTypeSize, true, TopKMode::TOPK_NORMAL, true,
        tilingData.topkTailTilingData);
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKSoftmaxV2KFullLoadTiling::GetTilingKey() const
{
    int sceneValue = 2;
    int doubleBufferFlag = 1;
    return TOPK_SOFTMAX_TILING_KEY_BASE_ALL + sceneValue * TOPK_SOFTMAX_TILING_KEY_BASE_SCENE +
           renorm * TOPK_SOFTMAX_TILING_KEY_BASE_RENORM + dtypeKey(dtype) * TOPK_SOFTMAX_TILING_KEY_BASE_DTYPE +
           doubleBufferFlag;
}

ge::graphStatus MoeGatingTopKSoftmaxV2KFullLoadTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYSTEM_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2KFullLoadTiling::PostTiling()
{
    tilingData.set_softmaxFlag(softmaxFlag);
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData.get_blockNum());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MoeGatingTopKSoftmaxV2", MoeGatingTopKSoftmaxV2KFullLoadTiling, 20000);
} // namespace optiling