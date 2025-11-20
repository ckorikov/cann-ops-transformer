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
 * \file grouped_matmul_swiglu_quant_v2_tiling.cpp
 * \brief
 */

#include <alog_pub.h>
#include <climits>
#include "log/log.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"
#include "register/op_impl_registry.h"
#include "grouped_matmul_swiglu_quant_v2_tiling.h"
using namespace Ops::Transformer::OpTiling;
using namespace GroupedMatmulSwigluQuantParamsV2;
using namespace optiling::GmmConstant;
namespace optiling{

void GroupedMatmulSwigluQuantDavidV2Tiling::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    return;
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
        inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
    }
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    return true;
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "xDesc is nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "wDesc is nullptr."), return false);
    inputParams_.bDtype = wDesc->GetDataType();
    auto scaleDesc = context_->GetInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr, OP_LOGE(context_->GetNodeName(), "scaleDesc is nullptr."), return false);
    inputParams_.scaleDtype = scaleDesc->GetDataType();
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    return CheckDtype();
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::CheckDtype()
{
    if ((inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2) &&
            (inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.bDtype == ge::DT_FLOAT8_E5M2)) {
        OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0 || inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                   OP_LOGE(
                       inputParams_.opName,
                                            "With DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2 inputs, \
            the expected dtype of xscale and weightscale should be DT_FLOAT8_E8M0, but actual dtype is %s, %s.",
                       ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                       ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
                   return false);
    } else {
        OP_LOGE(inputParams_.opName, "Quant case with x dtype %s and weight dtype %s is not supported.",
                  ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                  ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str());
        return false;
    }
    return true;
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::SetQuantModeForGMMSwigluQuant()
{
    if (IsMicroScaling()) {
        inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        return true;
    }
    return false;
}

bool GroupedMatmulSwigluQuantDavidV2Tiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "xStorageShape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    auto wStorageShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "wStorageShape is nullptr."), return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    auto scaleStorageShape = context_->GetInputShape(SCALE_INDEX);
    OP_CHECK_IF(scaleStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "scaleStorageShape is nullptr."), return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(scaleDimNum < 1,
               OP_LOGE(inputParams_.opName,
                                         "The dimension of xscale should be positive integer, actual is %zu",
                                         scaleDimNum),
               return false);
    auto x1ScaleStorageShape = context_->GetOptionalInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(x1ScaleStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "XScaleStorageShape is nullptr."), return false);
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."),
               return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    OP_CHECK_IF(!SetQuantModeForGMMSwigluQuant(),
               OP_LOGE(inputParams_.opName, "SetQuantModeForGMMSwigluQuant failed."), return false);
    return true;
}

ge::graphStatus GroupedMatmulSwigluQuantDavidV2Tiling::DoOpTiling()
{
    tilingData_.gmmSwigluQuantParams.set_groupNum(inputParams_.groupNum);
    tilingData_.gmmSwigluQuantParams.set_groupListType(static_cast<uint8_t>(inputParams_.groupListType));
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *dequantDtypeTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE);
        uint64_t dequantDtype = dequantDtypeTypePtr != nullptr ? static_cast<uint64_t>(*dequantDtypeTypePtr) : 0;
        const int64_t *quantDtypeTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE);
        uint64_t quantDtype = quantDtypeTypePtr != nullptr ? static_cast<uint64_t>(*quantDtypeTypePtr) : 0;
        tilingData_.gmmSwigluQuantParams.set_quantDtype(static_cast<uint8_t>(quantDtype));
    }
    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantDavidV2Tiling::DoLibApiTiling()
{
    CalBasicBlock();
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS,
               OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"), return ge::GRAPH_FAILED);
    auto baseM_modified = std::min(basicTiling_.baseM, static_cast<uint64_t>(128));
    tilingData_.mmTilingData.set_M(inputParams_.mSize);
    tilingData_.mmTilingData.set_N(inputParams_.nSize);
    tilingData_.mmTilingData.set_Ka(inputParams_.kSize);
    tilingData_.mmTilingData.set_Kb(inputParams_.kSize);
    tilingData_.mmTilingData.set_usedCoreNum(aicoreParams_.aicNum);
    tilingData_.mmTilingData.set_baseM(baseM_modified);
    tilingData_.mmTilingData.set_baseN(basicTiling_.baseN);
    tilingData_.mmTilingData.set_baseK(basicTiling_.baseK);
    tilingData_.mmTilingData.set_singleCoreM(baseM_modified);
    tilingData_.mmTilingData.set_singleCoreN(basicTiling_.singleCoreN);
    tilingData_.mmTilingData.set_singleCoreK(basicTiling_.singleCoreK);
    tilingData_.mmTilingData.set_depthA1(basicTiling_.depthA1);
    tilingData_.mmTilingData.set_depthB1(basicTiling_.depthB1);
    tilingData_.mmTilingData.set_stepM(basicTiling_.stepM);
    tilingData_.mmTilingData.set_stepN(basicTiling_.stepN);
    tilingData_.mmTilingData.set_stepKa(basicTiling_.stepKa);
    tilingData_.mmTilingData.set_stepKb(basicTiling_.stepKb);
    tilingData_.mmTilingData.set_isBias(inputParams_.hasBias ? 1 : 0);
    tilingData_.mmTilingData.set_iterateOrder(basicTiling_.iterateOrder);
    tilingData_.mmTilingData.set_dbL0A(2); // db switch, 1: off, 2: on
    tilingData_.mmTilingData.set_dbL0B(2); // db switch, 1: off, 2: on
    tilingData_.mmTilingData.set_dbL0C(basicTiling_.dbL0c);
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.mmTilingData.set_mxTypePara(
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA);
        } else {
            tilingData_.mmTilingData.set_mxTypePara(
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantDavidV2Tiling::PostTiling()
{
    context_->SetBlockDim(aicoreParams_.aicNum);
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
               OP_LOGE(context_->GetNodeName(), "Tiling data size[%zu] is not aligned to 8",
                                         tilingData_.GetDataSize()),
               return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void GroupedMatmulSwigluQuantDavidV2Tiling::PrintQuantParams()
{
    int32_t enable = AlogCheckDebugLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
    if (enable != 1) {
        return;
    }
    optiling::GMMSwigluQuantParams &params = tilingData_.gmmSwigluQuantParams;
    std::ostringstream oss;
    oss << "GMMQuantParams: groupNum = " << params.get_groupNum()
        << ", groupListType = " << static_cast<uint32_t>(params.get_groupListType())
        << ", quant_dtype = " << static_cast<int32_t>(params.get_quantDtype());
    OP_LOGD(inputParams_.opName, "%s", oss.str().c_str());
}

REGISTER_TILING_TEMPLATE("GroupedMatmulSwigluQuantV2", GroupedMatmulSwigluQuantDavidV2Tiling, 2);
} // namespace optiling