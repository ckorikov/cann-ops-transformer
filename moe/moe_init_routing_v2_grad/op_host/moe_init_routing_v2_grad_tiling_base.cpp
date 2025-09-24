/* *
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

/* !
 * \file moe_init_routing_v2_grad_tiling_base.cpp
 * \brief
 */
#include <cmath>
#include "moe_init_routing_v2_grad_tiling.h"
using namespace AscendC;
namespace optiling {
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static size_t DIM_THREE = 3;
const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aivNum = ascendcPlatform.GetCoreNumAiv();
    socVersion = ascendcPlatform.GetSocVersion();
    aicoreParams_.blockDim = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckDtypeValidity()
{
    const std::vector<ge::DataType> DATA_TYPE_SUPPORT = {
        ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};
    // 获取输入dtype
    auto gradExpandedXDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradExpandedXDesc);
    inDtype = gradExpandedXDesc->GetDataType();
    auto expandedRowIdxDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxDesc);
    auto rowIdxDtype = expandedRowIdxDesc->GetDataType();
    auto gradXDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXDesc);
    auto outDtype = gradXDesc->GetDataType();
    if (inDtype != outDtype) {
        OP_LOGE(context_->GetNodeName(), "inputdtype [%d] must be the same with outputDtype [%d].", inDtype, outDtype);
        return ge::GRAPH_FAILED;
    }
    if (rowIdxDtype != ge::DataType::DT_INT32) {
        OP_LOGE(context_->GetNodeName(), "gradExpandedX dtype only support [%d].", ge::DataType::DT_INT32);
        return ge::GRAPH_FAILED;
    }
    auto it = std::find(DATA_TYPE_SUPPORT.begin(), DATA_TYPE_SUPPORT.end(), inDtype);
    if (it == DATA_TYPE_SUPPORT.end()) {
        OP_LOGE(
            context_->GetNodeName(), "inputdtype [%d] must be in float32:[%d], float16:[%d], bfloat16:[%d]", inDtype,
            ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckShapeAllPositive(const gert::Shape& shape, std::string name)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) <= 0,
            OP_LOGE(
                context_->GetNodeName(), "Dim %lu of %s expect be positive, but actual %ld.", i, name.c_str(),
                shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckShapeValidity(
    const gert::Shape& xShape, const gert::Shape& rowIdxShape, const gert::Shape& gradXShape)
{
    if (CheckShapeAllPositive(rowIdxShape, "rowIdxShape") != ge::GRAPH_SUCCESS ||
        CheckShapeAllPositive(gradXShape, "gradXShape") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (dropPadMode == 0 && activeNum == 0) {
        if (CheckShapeAllPositive(xShape, "xShape") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        // 索引含-1场景，xShape可以有0
        for (size_t i = 0; i < xShape.GetDimNum(); i++) {
            OP_CHECK_IF(
                xShape.GetDim(i) < 0,
                OP_LOGE(
                    context_->GetNodeName(), "Dim %lu of x expect not be negtive, but actual %ld.", i,
                    xShape.GetDim(i)),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckParamsValidity(
    const gert::Shape& xShape, const gert::Shape& rowIdxShape, const gert::Shape& gradXShape) const
{
    // attr属性校验
    if (dropPadMode != 0 && dropPadMode != 1) {
        OP_LOGE(context_->GetNodeName(), "Attr drop_pad_mode should in range [0, 1].");
        return ge::GRAPH_FAILED;
    }

    if (topK <= 0) {
        OP_LOGE(context_->GetNodeName(), "Attr top_k must be larger than zero.");
        return ge::GRAPH_FAILED;
    }

    if (activeNum < 0) {
        OP_LOGE(context_->GetNodeName(), "Attr active_num should larger than or equal zero.");
        return ge::GRAPH_FAILED;
    }

    // shape校验
    size_t xDimNnum = xShape.GetDimNum();
    if (xDimNnum != DIM_TWO && xDimNnum != DIM_THREE) {
        OP_LOGE(context_->GetNodeName(), "The dim number of grad_expanded_x should be 2 or 3.");
        return ge::GRAPH_FAILED;
    }

    if (dropPadMode == 1 && xDimNnum != DIM_THREE) {
        OP_LOGE(context_->GetNodeName(), "GradExpandedX input shape must be 3D under Drop/Pad mode.");
        return ge::GRAPH_FAILED;
    }

    if (dropPadMode == 0) {
        if (activeNum == 0 && xShape.GetDim(0) != rowIdxShape.GetDim(0)) {
            OP_LOGE(context_->GetNodeName(), "All inputs Dim 0 size should be same under dropless mode.");
            return ge::GRAPH_FAILED;
        }

        if (activeNum > 0 && xShape.GetDim(0) != activeNum) {
            OP_LOGE(
                context_->GetNodeName(), "Dim 0 size of GradExpandedX should be equal active_num under active scene.");
            return ge::GRAPH_FAILED;
        }
    }

    size_t outDimNum = gradXShape.GetDimNum();
    if (outDimNum != DIM_TWO) {
        OP_LOGE(context_->GetNodeName(), "The dim number of grad_x should be 2.");
        return ge::GRAPH_FAILED;
    }

    int64_t hiddenSize = (dropPadMode == 0) ? xShape.GetDim(1) : xShape.GetDim(2); // 2: drop/pad 场景，尾轴在第三维
    if (gradXShape.GetDim(1) != hiddenSize) {
        OP_LOGE(context_->GetNodeName(), "Tail dim size of input and output should be same.");
        return ge::GRAPH_FAILED;
    }

    if (gradXShape.GetDim(0) != rowIdxShape.GetDim(0) / topK) {
        OP_LOGE(
            context_->GetNodeName(),
            "output shape invalid, dim 0 of output gradX should be equal to the division of expandedRowIdx dim 0 and "
            "topK.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();

    // 获取输入shape
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    const gert::Shape xShape = xShapePtr->GetStorageShape();
    auto rowIdxShapePtr = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rowIdxShapePtr);
    const gert::Shape rowIdxShape = rowIdxShapePtr->GetStorageShape();
    auto gradXShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXShapePtr);
    const gert::Shape gradXShape = gradXShapePtr->GetStorageShape();

    // 获取输入属性
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* topKPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, topKPtr);
    topK = *topKPtr;
    const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(1); // 1: drop_pad_mode attr
    dropPadMode = (dropPadModePtr == nullptr) ? 0 : *dropPadModePtr;
    const int64_t* activeNumPtr = attrs->GetAttrPointer<int64_t>(2); // 2: active_num attr
    activeNum = (activeNumPtr == nullptr) ? 0 : *activeNumPtr;

    ge::graphStatus res = CheckDtypeValidity();
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }

    // 参数校验
    res = CheckParamsValidity(xShape, rowIdxShape, gradXShape);
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }

    // shape校验
    if (CheckShapeValidity(xShape, rowIdxShape, gradXShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 设置tilingData基本参数
    int64_t BSK = rowIdxShape.GetDim(0); // A: B*S*K
    E = (dropPadMode == 0) ? 0 : xShape.GetDim(0);
    C = (dropPadMode == 0) ? 0 : xShape.GetDim(1);
    hiddenSize = (dropPadMode == 0) ? xShape.GetDim(1) : xShape.GetDim(2); // 2 for H, when shape is {E, C, H}
    if (BSK % topK != 0) {
        OP_LOGE(context_->GetNodeName(), "expanded_row_idx shape or top_k val invalid");
        return ge::GRAPH_FAILED;
    }
    N = BSK / topK;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::GetWorkspaceSize()
{
    // 计算workspace大小，无需workspace临时空间，不存在多核同步，预留固定大小即可
    workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMoeInitRoutingV2Grad(gert::TilingContext* context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForMoeInitRoutingV2Grad(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MoeInitRoutingV2Grad)
    .Tiling(TilingForMoeInitRoutingV2Grad)
    .TilingParse<MoeInitRoutingV2GradCompileInfo>(TilingPrepareForMoeInitRoutingV2Grad);
} // namespace optiling