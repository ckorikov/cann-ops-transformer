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
 * \file quant_reduce_scatter_check_tiling.cpp
 * \brief
 */
#include "quant_reduce_scatter_check_tiling.h"

namespace MC2Tiling {
constexpr size_t X_INDEX = 0;
constexpr size_t SCALE_INDEX = 1;
constexpr size_t OUTPUT_INDEX = 0;
constexpr size_t GROUP_INDEX = 0;
constexpr size_t REDUCE_OP_INDEX = 1;
constexpr size_t OUT_PUT_DTYPE_INDEX = 2;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
const char *REDUCE_OP_TYPE = "sum";
constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t THREE_DIMS = 3;
constexpr uint32_t NO_MATCH_QUANT_MOD = 0;
constexpr uint32_t TG_QUANT_MOD = 1;
constexpr uint32_t MX_QUANT_MOD = 2;
constexpr uint64_t TG_QUANT_NUMBER = 128UL;
constexpr uint64_t MX_QUANT_NUMBER = 64UL;
constexpr uint64_t MX_SCALE_LAST_DIM = 2UL;
constexpr uint32_t X_DTYPE_SIZE_ONE = 1;
constexpr uint32_t SCALE_DTYPE_SIZE_ONE = 1;
constexpr uint32_t SCALE_DTYPE_SIZE_FOUR = 4;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t DEFAULT_WIN_SIZE = 0UL;
constexpr uint64_t H_VALUE_ONE = 5120UL;
constexpr uint64_t H_VALUE_TWO = 7168UL;
constexpr size_t NUM_THREE = 3;

ge::graphStatus QuantReduceScatterCheckTiling::CheckAttrs(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    // group空字符串校验
    const char *groupPtr = attrs->GetAttrPointer<char>(GROUP_INDEX);
    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE(nodeName, "groupPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::string(groupPtr).empty(),
        OP_LOGE(nodeName, "group should not be empty."), return ge::GRAPH_FAILED);
    // reducetype校验是否为sum
    const char *reduceOpPtr = attrs->GetAttrPointer<char>(REDUCE_OP_INDEX);
    OP_TILING_CHECK(reduceOpPtr == nullptr, OP_LOGE(nodeName, "reduceOpPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(strncmp(reduceOpPtr, REDUCE_OP_TYPE, NUM_THREE) != 0,
        OP_LOGE(nodeName, "reduce_op is invalid, the type should be %s, but current reduce_op is %s.",
        REDUCE_OP_TYPE, reduceOpPtr), return ge::GRAPH_FAILED);
    // outPutType校验
    const int64_t *outPutTypePtr = attrs->GetAttrPointer<int64_t>(OUT_PUT_DTYPE_INDEX);
    OP_TILING_CHECK(outPutTypePtr == nullptr, OP_LOGE(nodeName, "outPutTypePtr is nullptr."), return ge::GRAPH_FAILED);
    ge::DataType outPutType = static_cast<ge::DataType>(*outPutTypePtr);
    OP_TILING_CHECK((outPutType != ge::DT_BF16) && (outPutType != ge::DT_FLOAT) && (outPutType != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "outPutType is invalid, outPutType should be bfloat16 or float or float16, but is %s.",
        Ops::Base::ToString(outPutType).c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool QuantReduceScatterCheckTiling::CheckDimWithQuantMode(const gert::TilingContext *context,
                                                          const QuantReduceScatterTilingParams &params)
{
    const char *nodeName = context->GetNodeName();
    uint32_t quantMode = params.quantMode;
    size_t scaleDim = context->GetInputShape(SCALE_INDEX)->GetStorageShape().GetDimNum();
    uint64_t xValueTwo = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(1);
    uint64_t scaleValueTwo = context->GetInputShape(SCALE_INDEX)->GetStorageShape().GetDim(1);
    // 根据量化模式，校验scale，TG：scale(bs, H/128)，MX：scale(bs, H/64, 2)
    if (quantMode == TG_QUANT_MOD) {
        OP_TILING_CHECK(scaleDim != TWO_DIMS, OP_LOGE(nodeName, "In the TG quantmode, scaleDim should be 2, "
            "but current scaleDim is %lu.", scaleDim), return false);
        OP_TILING_CHECK(ops::CeilDiv(xValueTwo, TG_QUANT_NUMBER) != scaleValueTwo,
            OP_LOGE(nodeName, "In the TG quantmode, scale dim1 should be equal to x dim1 divided by 128, "
            "but x dim1 is %lu, scale dim1 is %lu.", xValueTwo, scaleValueTwo), return false);
    } else if (quantMode == MX_QUANT_MOD) {
        OP_TILING_CHECK(scaleDim != THREE_DIMS, OP_LOGE(nodeName, "In the MX quantmode, scaleDim should be 3, "
            "but current scaleDim is %lu.", scaleDim), return false);
        uint64_t scaleValueThree = context->GetInputShape(SCALE_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        // 校验是否为空tensor
        OP_TILING_CHECK(scaleValueThree == 0,
            OP_LOGE(nodeName, "scale should not be empty tensor."), return false);
        OP_TILING_CHECK(ops::CeilDiv(xValueTwo, MX_QUANT_NUMBER) != scaleValueTwo,
            OP_LOGE(nodeName, "In the MX quantmode, scale dim1 should be equal to x dim1 divided by 64, "
            "but x dim1 is %lu, scale dim1 is %lu.", xValueTwo, scaleValueTwo), return false);
        OP_TILING_CHECK(scaleValueThree != MX_SCALE_LAST_DIM, OP_LOGE(nodeName, "In the MX quantmode, The scale "
            "dim2 is invalid, scale dim2 should be 2, but is %lu.", scaleValueThree), return false);
    }
    return true;
}

bool QuantReduceScatterCheckTiling::CheckInputTensorDim(const gert::TilingContext *context,
                                                        const QuantReduceScatterTilingParams &params)
{
    const char *nodeName = context->GetNodeName();
    // 输入x的(bs,H),H需要为128的倍数
    const gert::StorageShape *xShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    const gert::StorageShape *scaleShape = context->GetInputShape(SCALE_INDEX);
    OP_TILING_CHECK(scaleShape == nullptr, OP_LOGE(nodeName, "scaleShape is null."), return false);
    // xDim只能为二维，scale的Dim根据量化模式判断
    size_t xDim = xShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(xDim != TWO_DIMS,
        OP_LOGE(nodeName, "xDim should be 2, but current xDim is %lu.", xDim), return false);
    uint64_t xValueOne = xShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = xShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t scaleValueOne = scaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t scaleValueTwo = scaleShape->GetStorageShape().GetDim(DIM_ONE);
    // 校验是否为空tensor
    OP_TILING_CHECK((xValueOne == 0) || (xValueTwo == 0) || (scaleValueOne == 0) || (scaleValueTwo == 0),
            OP_LOGE(nodeName, "x and scale should not be empty tensor."), return false);
    // 校验第一维
    OP_TILING_CHECK(xValueOne != scaleValueOne, OP_LOGE(nodeName, "The first dimension of scale %lu is not equal to "
        "x %lu which is invalid.", scaleValueOne, xValueOne), return false);
    uint32_t rankSize = params.rankSize;
    OP_TILING_CHECK(xValueOne % rankSize != 0, OP_LOGE(nodeName, "The first dimension of x should be multiple of "
        "ranksize, but x is %lu, ranksize is %u.", xValueOne, rankSize), return false);
    // 校验第二维，需为5120和7168，泛化场景则H为128倍数
    OP_TILING_CHECK(xValueTwo % TG_QUANT_NUMBER != 0, OP_LOGE(nodeName, "The last dimension of x is invalid, x "
        "last dimension should be multiple of 128, but is %lu.", xValueTwo), return false);
    OP_TILING_CHECK((xValueTwo != H_VALUE_ONE) && (xValueTwo != H_VALUE_TWO), OP_LOGE(nodeName, "The last dimension "
        "of x is invalid, x last dimension should be 5120 or 7168, but is %lu.", xValueTwo), return false);
    OP_TILING_CHECK(!CheckDimWithQuantMode(context, params),
        OP_LOGE(nodeName, "The dimensions of x and scale are invalid in the quantmode."), return false);
    return true;
}

bool QuantReduceScatterCheckTiling::CheckOutputTensorDim(const gert::TilingContext *context,
                                                         const QuantReduceScatterTilingParams &params)
{
    const char *nodeName = context->GetNodeName();
    // output(bs/rankNum, H)，output必须为2维
    const gert::StorageShape *outputShape = context->GetOutputShape(OUTPUT_INDEX);
    OP_TILING_CHECK(outputShape == nullptr, OP_LOGE(nodeName, "The outputShape is null."), return false);
    OP_TILING_CHECK(outputShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "outputShape must be 2, but current dim num is %lu",
                outputShape->GetStorageShape().GetDimNum()), return false);
    uint32_t rankSize = params.rankSize;
    uint64_t outputValueOne = outputShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t outputValueTwo = outputShape->GetStorageShape().GetDim(DIM_ONE);
    // 之前checkInputTensorDim对x的合法性校验过，此处跳过
    uint64_t xValueOne = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    // 检查x和output维度关系
    OP_TILING_CHECK(xValueOne / rankSize != outputValueOne,
        OP_LOGE(nodeName, "The output dim0 is invalid, output dim0 should be the dim0 of x divided by rankSize,"
        "but output dim0 is %lu, x dim0 is %lu, rankSize is %u.", outputValueOne, xValueOne, rankSize), return false);
    OP_TILING_CHECK(xValueTwo != outputValueTwo,
        OP_LOGE(nodeName, "The output dim1 is invalid, output dim1 should be the dim0 of x divided by rankSize,"
        "but output dim0 is %lu, x dim0 is %lu, rankSize is %u.", outputValueTwo, xValueTwo, rankSize), return false);
    return true;
}

bool QuantReduceScatterCheckTiling::CheckInputDtypeAndSetQuantMode(const gert::TilingContext *context,
                                                                   QuantReduceScatterTilingParams &params)
{
    const char *nodeName = context->GetNodeName();
    ge::DataType xDtype = context->GetInputDesc(X_INDEX)->GetDataType();
    ge::DataType scaleDtype = context->GetInputDesc(SCALE_INDEX)->GetDataType();
    OP_TILING_CHECK((xDtype != ge::DT_INT8) && (xDtype != ge::DT_HIFLOAT8) && (xDtype != ge::DT_FLOAT8_E4M3FN) &&
                    (xDtype != ge::DT_FLOAT8_E5M2),
        OP_LOGE(nodeName, "x dataType should be int8 or hifloat8 or float8_e4m3fn or float8_e5m2, but is %s.",
        Ops::Base::ToString(xDtype).c_str()), return false);
    OP_TILING_CHECK((scaleDtype != ge::DT_FLOAT) && (scaleDtype != ge::DT_FLOAT8_E8M0),
        OP_LOGE(nodeName, "scale dataType is invalid, dataType should be float or float8_e8m0, but is %s.",
        Ops::Base::ToString(scaleDtype).c_str()), return false);
    uint32_t quantMode = NO_MATCH_QUANT_MOD;
    if (((xDtype == ge::DT_INT8) || (xDtype == ge::DT_HIFLOAT8) || (xDtype == ge::DT_FLOAT8_E4M3FN) ||
         (xDtype == ge::DT_FLOAT8_E5M2)) && (scaleDtype == ge::DT_FLOAT)) {
        quantMode = TG_QUANT_MOD;
    } else if (((xDtype == ge::DT_FLOAT8_E4M3FN) || (xDtype == ge::DT_FLOAT8_E5M2)) &&
               (scaleDtype == ge::DT_FLOAT8_E8M0)) {
        quantMode = MX_QUANT_MOD;
    }
    // quantMode为0：无量化模式匹配，为1：TG量化，为2：MX量化
    OP_TILING_CHECK(!static_cast<bool>(quantMode),
        OP_LOGE(nodeName, "The x dataType %s and scale dataType %s do not match any quantMode.",
        Ops::Base::ToString(xDtype).c_str(), Ops::Base::ToString(scaleDtype).c_str()), return false);
    params.quantMode = quantMode;
    return true;
}

bool QuantReduceScatterCheckTiling::CheckTensorDataType(const gert::TilingContext *context,
                                                        QuantReduceScatterTilingParams &params)
{
    const char *nodeName = context->GetNodeName();
    // 检查输入x和scale的dtype
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    auto scaleDesc = context->GetInputDesc(SCALE_INDEX);
    OP_TILING_CHECK(scaleDesc == nullptr, OP_LOGE(nodeName, "scaleDesc is null"), return false);
    OP_TILING_CHECK(!CheckInputDtypeAndSetQuantMode(context, params),
        OP_LOGE(nodeName, "The dtype of input params are invalid."), return false);
    // 检查output的Dtype
    auto outputDesc = context->GetOutputDesc(OUTPUT_INDEX);
    OP_TILING_CHECK(outputDesc == nullptr, OP_LOGE(nodeName, "OutputDesc is null."), return false);
    ge::DataType outputType = outputDesc->GetDataType();
    OP_TILING_CHECK(((outputType != ge::DT_FLOAT16) && (outputType != ge::DT_BF16) && (outputType != ge::DT_FLOAT)),
        OP_LOGE(nodeName, "output dataType is invalid, dataType should be float16 or bfloat16 or float, but is %s.",
        Ops::Base::ToString(outputType).c_str()), return false);
    return true;
}

bool QuantReduceScatterCheckTiling::CheckTensorDim(const gert::TilingContext *context,
                                                   const QuantReduceScatterTilingParams &params)
{
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(!CheckInputTensorDim(context, params),
        OP_LOGE(nodeName, "Input param shape is invalid."), return false);
    OP_TILING_CHECK(!CheckOutputTensorDim(context, params),
        OP_LOGE(nodeName, "Output param shape is invalid."), return false);
    return true;
}

bool QuantReduceScatterCheckTiling::CheckTensorFormat(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto xDesc = context->GetInputDesc(X_INDEX);
    ge::Format xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()));
    OP_TILING_CHECK(xFormat != ge::FORMAT_ND,
        OP_LOGE(nodeName, "x format is invalid, format should be NZ, but is %s.",
        Ops::Base::ToString(xFormat).c_str()), return false);
    
    auto scaleDesc = context->GetInputDesc(SCALE_INDEX);
    ge::Format scaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(scaleDesc->GetStorageFormat()));
    OP_TILING_CHECK(scaleFormat != ge::FORMAT_ND,
        OP_LOGE(nodeName, "scale format is invalid, format should be NZ, but is %s.",
        Ops::Base::ToString(scaleFormat).c_str()), return false);

    auto outputDesc = context->GetOutputDesc(OUTPUT_INDEX);
    ge::Format outPutFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(outputDesc->GetStorageFormat()));
    OP_TILING_CHECK(outPutFormat != ge::FORMAT_ND,
        OP_LOGE(nodeName, "output format is invalid, format should be NZ, but is %s.",
        Ops::Base::ToString(outPutFormat).c_str()), return false);
    return true;
}

bool QuantReduceScatterCheckTiling::CheckWindowSize(const gert::TilingContext *context,
                                                    const QuantReduceScatterTilingParams &params)
{
    const char *nodeName = context->GetNodeName();
    // 获取量化模式，获取数据类型
    uint32_t quantMode = params.quantMode;
    uint64_t xValueOne = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t scaleValueOne = context->GetInputShape(SCALE_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t scaleValueTwo = context->GetInputShape(SCALE_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t xDataSize = ((xValueOne * xValueTwo * X_DTYPE_SIZE_ONE + WIN_ADDR_ALIGN - 1UL) /
                           WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    uint64_t actulWinSize = DEFAULT_WIN_SIZE;
    if (quantMode == TG_QUANT_MOD) {
        uint64_t scaleSize = scaleValueOne * scaleValueTwo * SCALE_DTYPE_SIZE_FOUR;
        uint64_t scaleDataSize = ((scaleSize + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
        // 数据区（x和scale）+状态区（1Mb）
        actulWinSize = xDataSize + scaleDataSize + MB_SIZE;
    } else if (quantMode == MX_QUANT_MOD) {
        uint64_t scaleValueThree = context->GetInputShape(SCALE_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        uint64_t scaleSize = scaleValueOne * scaleValueTwo * scaleValueThree * SCALE_DTYPE_SIZE_ONE;
        uint64_t scaleDataSize = ((scaleSize + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
        actulWinSize = xDataSize + scaleDataSize + MB_SIZE;
    }
    uint64_t maxWindowSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    OP_TILING_CHECK(actulWinSize > maxWindowSize,
        OP_LOGE(nodeName, "actulWinSize %lu is bigger than maxWindowSize %lu which is invalid.",
        actulWinSize, maxWindowSize), return false);
    return true;
}

ge::graphStatus QuantReduceScatterCheckTiling::TilingCheckQuantReduceScatter(const gert::TilingContext *context,
                                                                             QuantReduceScatterTilingParams &params)
{
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(!CheckTensorDataType(context, params),
        OP_LOGE(nodeName, "params dtype is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDim(context, params),
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context),
        OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace MC2Tiling
