/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_torch.cpp
 * \brief
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"
#include <torch/torch.h>
#include <ATen/Dispatch.h>


#include "kernel_operator.h"


// 直接调用math目录中已经实现的算子公共逻辑
// #include "math/is_finite/op_kernel/is_finite.h"
// #include "math/is_finite/op_host/is_finite_tiling_common.h"
#include "gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling_common.h"
#include "grouped_matmul_torch.h"

namespace ascend_ops {

namespace GroupedMatmul {

using namespace GroupedMatmulNs;

// using namespace GroupedMatmulNs;
// 只传输入输出tensor，attr在tilingData里
template <typename xT, typename weightT, typename biasT, typename scaleT, typename offsetT, typename antiquantScaleT,
          typename antiquantOffsetT, typename groupListT, typename perTokenScaleT, typename yT>
__global__ __aicore__ void groupedmatmul_kernel(__gm__ uint8_t *x, __gm__ uint8_t *weight, __gm__ uint8_t *bias,
                                                __gm__ uint8_t *scale, __gm__ uint8_t *offset,
                                                __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                                __gm__ uint8_t *groupList, __gm__ uint8_t *perTokenScale,
                                                __gm__ uint8_t *y, const GMMTilingData tilingData)
{
    GroupedMatmulKernelImpl<D_T_A, D_T_B, D_T_Y, TRANS_A, TRANS_B, GROUP_LIST_TYPE, IS_STATIC_TILING_API,
                            A8W4_KERNEL_TEMPLATE, A16W8_KERNEL_TEMPLATE, AIV_AIC_RATIO>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, y, &tilingData);
    return;
}

// 传输入输出tensor和attr
template <typename xT, typename weightT, typename biasT, typename scaleT, typename offsetT, typename antiquantScaleT,
          typename antiquantOffsetT, typename groupListT, typename perTokenScaleT, typename yT>
void groupedmatmul_api(aclrtStream stream, const at::TensorList &x, const at::TensorList &weight,
                       const c10::optional<at::TensorList> &bias, const c10::optional<at::TensorList> &scale,
                       const c10::optional<at::TensorList> &offset, const c10::optional<at::TensorList> &antiquantScale,
                       const c10::optional<at::TensorList> &antiquantOffset,
                       const c10::optional<torch::Tensor> &groupList,
                       const c10::optional<at::TensorList> &perTokenScale, const at::TensorList &y,
                       const int64_t splitItem, const int64_t groupType, const int64_t groupListType,
                       const int64_t actType, const vector<int64_t> *tuningConfigOptional)
{
    // int64_t num_element = x.numel();
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizePlatFrom;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatFrom);
    GMMTilingData tilingData;
    GroupedMatmulNs::GroupedMatmulTiling::GroupedMatmulCommonTiling<at::TensorList, c10::optional<at::TensorList>,
                                                                    c10::optional<torch::Tensor>>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, tilingData,
        ascendcPlatform->GetCoreNumAiv(), ubSizePlatFrom);
    uint32_t blockDim = tilingData.needCoreNum;
    // 必填参数：不允许为空
    auto x_ptr = get_first_tensor_address<at::TensorList, xT>(x, false);
    auto weight_ptr = get_first_tensor_address<at::TensorList, weightT>(weight, false);
    auto y_ptr = get_first_tensor_address<at::TensorList, yT>(y, false);

    // 可选参数：允许为空
    auto bias_ptr = get_first_tensor_address<c10::optional<at::TensorList>, biasT>(bias, true);
    auto scale_ptr = get_first_tensor_address<c10::optional<at::TensorList>, scaleT>(scale, true);
    auto offset_ptr = get_first_tensor_address<c10::optional<at::TensorList>, offsetT>(offset, true);
    auto antiquantScale_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>, antiquantScaleT>(antiquantScale, true);
    auto antiquantOffset_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>, antiquantOffsetT>(antiquantOffset, true);
    auto groupList_ptr = get_first_tensor_address<c10::optional<torch::Tensor>, groupListT>(groupList, true);
    auto perTokenScale_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>, perTokenScaleT>(perTokenScale, true);
    groupedmatmul_kernel<xT, weightT, biasT, scaleT, offsetT, antiquantScaleT, antiquantOffsetT, groupListT,
                         perTokenScaleT, yT><<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t *)x_ptr, (__gm__ uint8_t *)weight_ptr, (__gm__ uint8_t *)bias_ptr, (__gm__ uint8_t *)scale_ptr,
        (__gm__ uint8_t *)offset_ptr, (__gm__ uint8_t *)antiquantScale_ptr, (__gm__ uint8_t *)antiquantOffset_ptr,
        (__gm__ uint8_t *)groupList_ptr, (__gm__ uint8_t *)perTokenScale_ptr, (__gm__ uint8_t *)y_ptr, tilingData);
}

// 定义类型列表（全局静态，编译时初始化）
namespace {
// 每个输入的支持数据类型列表
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_X = {
    at::kHalf, at::kBFloat16, at::kChar,     at::kHalf, at::kBFloat16, at::kFloat, at::kChar, at::kChar, at::kChar,
    at::kChar, at::kHalf,     at::kBFloat16, at::kHalf, at::kBFloat16, at::kChar,  at::kChar, at::kChar, at::kChar,
    at::kChar, at::kChar,     at::kInt,      at::kInt,  at::kInt,      at::kInt,   at::kChar, at::kChar};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_WEIGHT = {
    at::kHalf, at::kBFloat16, at::kChar, at::kChar, at::kChar,     at::kFloat, at::kChar, at::kChar, at::kChar,
    at::kChar, at::kInt,      at::kInt,  at::kHalf, at::kBFloat16, at::kChar,  at::kChar, at::kInt,  at::kInt,
    at::kInt,  at::kInt,      at::kInt,  at::kInt,  at::kInt,      at::kInt,   at::kChar, at::kChar};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_BIAS = {
    at::kHalf,  at::kFloat, at::kInt,   at::kHalf, at::kFloat, at::kFloat, at::kInt,      at::kInt,     at::kInt,
    at::kInt,   at::kHalf,  at::kFloat, at::kHalf, at::kFloat, at::kInt,   at::kInt,      at::kFloat,   at::kFloat,
    at::kFloat, at::kFloat, at::kHalf,  at::kHalf, at::kHalf,  at::kHalf,  at::kBFloat16, at::kBFloat16};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_SCALE = {
    at::kLong,  at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kBFloat16, at::kBFloat16, at::kFloat,
    at::kFloat, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong,     at::kLong,     at::kLong,
    at::kLong,  at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kBFloat16, at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_OFFSET = {
    at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat,
    at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat,
    at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_ANTIQUANTSCALE = {
    at::kHalf, at::kHalf, at::kHalf,     at::kHalf, at::kBFloat16, at::kHalf, at::kHalf, at::kHalf, at::kHalf,
    at::kHalf, at::kHalf, at::kBFloat16, at::kHalf, at::kHalf,     at::kHalf, at::kHalf, at::kHalf, at::kHalf,
    at::kHalf, at::kHalf, at::kHalf,     at::kHalf, at::kHalf,     at::kHalf, at::kHalf, at::kHalf};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_ANTIQUANTOFFSET = {
    at::kHalf, at::kHalf, at::kHalf,     at::kHalf, at::kBFloat16, at::kHalf, at::kHalf, at::kHalf, at::kHalf,
    at::kHalf, at::kHalf, at::kBFloat16, at::kHalf, at::kHalf,     at::kHalf, at::kHalf, at::kHalf, at::kHalf,
    at::kHalf, at::kHalf, at::kHalf,     at::kHalf, at::kHalf,     at::kHalf, at::kHalf, at::kHalf};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_GROUPLIST = {
    at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong,
    at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong,
    at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kLong};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_PERTOKENSCALE = {
    at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat,
    at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat,
    at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat, at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_OUTPUT = {
    at::kHalf,     at::kBFloat16, at::kChar,     at::kHalf,     at::kBFloat16, at::kFloat, at::kBFloat16,
    at::kBFloat16, at::kHalf,     at::kHalf,     at::kHalf,     at::kBFloat16, at::kHalf,  at::kBFloat16,
    at::kInt,      at::kInt,      at::kBFloat16, at::kHalf,     at::kBFloat16, at::kHalf,  at::kHalf,
    at::kBFloat16, at::kHalf,     at::kBFloat16, at::kBFloat16, at::kBFloat16};

// 组合缓存（使用函数返回静态引用确保初始化顺序正确）
const std::vector<TypeCombo> &getSupportedCombos()
{
    static const auto combos = TypeComboManager::createCombosFromLists(
        SUPPORTED_DTYPE_X, SUPPORTED_DTYPE_BIAS, SUPPORTED_DTYPE_SCALE, SUPPORTED_DTYPE_OFFSET,
        SUPPORTED_DTYPE_ANTIQUANTSCALE, SUPPORTED_DTYPE_ANTIQUANTOFFSET, SUPPORTED_DTYPE_GROUPLIST,
        SUPPORTED_DTYPE_PERTOKENSCALE, SUPPORTED_DTYPE_WEIGHT, SUPPORTED_DTYPE_OUTPUT);
    return combos;
}
} // namespace


// 关键修正：
// 1. 宏内参数列表合并为单行，移除所有换行和行内注释（避免语法解析错误）
// 2. 确保模板参数/函数参数括号匹配，无多余逗号
#define CALL_GROUPEDMATMUL(X_ID, WEIGHT_ID, BIAS_ID, SCALE_ID, OFFSET_ID, ANTIQ_SCALE_ID, ANTIQ_OFFSET_ID,             \
                           GROUP_LIST_ID, PER_TOKEN_SCALE_ID, OUTPUT_ID)                                               \
    groupedmatmul_api<IdToCppType_t<X_ID>, IdToCppType_t<WEIGHT_ID>, IdToCppType_t<BIAS_ID>, IdToCppType_t<SCALE_ID>,  \
                      IdToCppType_t<OFFSET_ID>, IdToCppType_t<ANTIQ_SCALE_ID>, IdToCppType_t<ANTIQ_OFFSET_ID>,         \
                      IdToCppType_t<GROUP_LIST_ID>, IdToCppType_t<PER_TOKEN_SCALE_ID>, IdToCppType_t<OUTPUT_ID>>(      \
        stream, x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, y,          \
        splitItem, groupType, groupListType, actType, tuningConfigOptional);
// ===================== 第三步：重写 dispatch_groupedmatmul（枚举有效组合） =====================
void dispatch_groupedmatmul(const TypeCombo &matched_combo,
                            aclrtStream stream, // 注意：stream 入参位置提前，和函数定义一致
                            const torch::TensorList &x, const torch::TensorList &weight,
                            const c10::optional<torch::TensorList> &bias, const c10::optional<torch::TensorList> &scale,
                            const c10::optional<torch::TensorList> &offset,
                            const c10::optional<torch::TensorList> &antiquantScale,
                            const c10::optional<torch::TensorList> &antiquantOffset,
                            const c10::optional<torch::Tensor> &groupList,
                            const c10::optional<torch::TensorList> &perTokenScale,
                            const at::TensorList &y, // 新增：补充漏掉的 y 参数（函数定义必传）
                            const int64_t splitItem, const int64_t groupType, const int64_t groupListType,
                            const int64_t actType, const vector<int64_t> *tuningConfigOptional)
{
    // 1. 将 matched_combo 中的 at::ScalarType 转换为 ScalarTypeId（整型 ID）
    const auto x_id = getScalarTypeId(matched_combo.x);
    const auto weight_id = getScalarTypeId(matched_combo.weight);
    const auto bias_id = getScalarTypeId(matched_combo.bias);
    const auto scale_id = getScalarTypeId(matched_combo.scale);
    const auto offset_id = getScalarTypeId(matched_combo.offset);
    const auto antiq_scale_id = getScalarTypeId(matched_combo.antiquantScale);
    const auto antiq_offset_id = getScalarTypeId(matched_combo.antiquantOffset);
    const auto group_list_id = getScalarTypeId(matched_combo.groupList);
    const auto per_token_scale_id = getScalarTypeId(matched_combo.perTokenScale);
    const auto output_id = getScalarTypeId(matched_combo.output);

    // 2. 枚举所有支持的类型组合（仅十几种，避免编译器爆炸）
    // 组合 1：x=HALF(0), weight=HALF(0), bias=FLOAT(3), ..., output=HALF(0)
    if (x_id == ScalarTypeId::HALF && weight_id == ScalarTypeId::HALF && bias_id == ScalarTypeId::FLOAT &&
        scale_id == ScalarTypeId::HALF && offset_id == ScalarTypeId::HALF && antiq_scale_id == ScalarTypeId::FLOAT &&
        antiq_offset_id == ScalarTypeId::FLOAT && group_list_id == ScalarTypeId::INT &&
        per_token_scale_id == ScalarTypeId::FLOAT && output_id == ScalarTypeId::HALF) {
        CALL_GROUPEDMATMUL(ScalarTypeId::HALF, ScalarTypeId::HALF, ScalarTypeId::FLOAT, ScalarTypeId::HALF,
                           ScalarTypeId::HALF, ScalarTypeId::FLOAT, ScalarTypeId::FLOAT, ScalarTypeId::INT,
                           ScalarTypeId::FLOAT,
                           ScalarTypeId::HALF); // 10 个模板参数，宏内部补全 16 个函数参数
    }
    // // 组合 2：x=BFLOAT16(1), weight=BFLOAT16(1), bias=FLOAT(3), ..., output=BFLOAT16(1)
    // else if (x_id == ScalarTypeId::BFLOAT16 && weight_id == ScalarTypeId::BFLOAT16 &&
    //          bias_id == ScalarTypeId::FLOAT && scale_id == ScalarTypeId::BFLOAT16 &&
    //          offset_id == ScalarTypeId::BFLOAT16 && antiq_scale_id == ScalarTypeId::FLOAT &&
    //          antiq_offset_id == ScalarTypeId::FLOAT && group_list_id == ScalarTypeId::INT &&
    //          per_token_scale_id == ScalarTypeId::FLOAT && output_id == ScalarTypeId::BFLOAT16) {
    //     CALL_GROUPEDMATMUL(ScalarTypeId::BFLOAT16, ScalarTypeId::BFLOAT16, ScalarTypeId::FLOAT,
    //                        ScalarTypeId::BFLOAT16, ScalarTypeId::BFLOAT16, ScalarTypeId::FLOAT,
    //                        ScalarTypeId::FLOAT, ScalarTypeId::INT, ScalarTypeId::FLOAT,
    //                        ScalarTypeId::BFLOAT16);
    // }
    // // 组合 3：x=CHAR(2), weight=CHAR(2), bias=FLOAT(3), ..., output=CHAR(2)
    // else if (x_id == ScalarTypeId::CHAR && weight_id == ScalarTypeId::CHAR &&
    //          bias_id == ScalarTypeId::FLOAT && scale_id == ScalarTypeId::FLOAT &&
    //          offset_id == ScalarTypeId::FLOAT && antiq_scale_id == ScalarTypeId::FLOAT &&
    //          antiq_offset_id == ScalarTypeId::FLOAT && group_list_id == ScalarTypeId::INT &&
    //          per_token_scale_id == ScalarTypeId::FLOAT && output_id == ScalarTypeId::CHAR) {
    //     CALL_GROUPEDMATMUL(ScalarTypeId::CHAR, ScalarTypeId::CHAR, ScalarTypeId::FLOAT,
    //                        ScalarTypeId::FLOAT, ScalarTypeId::FLOAT, ScalarTypeId::FLOAT,
    //                        ScalarTypeId::FLOAT, ScalarTypeId::INT, ScalarTypeId::FLOAT,
    //                        ScalarTypeId::CHAR);
    // }
    // ========== 补充你实际支持的其他组合（总共十几种） ==========
    // else if (...) { ... }

    // 无效组合：抛出明确错误
    else {
        TORCH_CHECK(
            false, "Unsupported type combo for groupedmatmul: \n", "x=", static_cast<int>(x_id),
            ", weight=", static_cast<int>(weight_id), ", bias=", static_cast<int>(bias_id), "\n",
            "scale=", static_cast<int>(scale_id), ", offset=", static_cast<int>(offset_id),
            ", antiq_scale=", static_cast<int>(antiq_scale_id), "\n",
            "antiq_offset=", static_cast<int>(antiq_offset_id), ", group_list=", static_cast<int>(group_list_id),
            ", per_token_scale=", static_cast<int>(per_token_scale_id), "\n", "output=", static_cast<int>(output_id));
    }
}

// 传输入tensor和attr
torch::Tensor groupedmatmul_npu(
    const torch::TensorList &x, const torch::TensorList &weight, const c10::optional<torch::TensorList> &bias,
    const c10::optional<torch::TensorList> &scale, const c10::optional<torch::TensorList> &offset,
    const c10::optional<torch::TensorList> &antiquantScale, const c10::optional<torch::TensorList> &antiquantOffset,
    const c10::optional<torch::Tensor> &groupList, const c10::optional<torch::TensorList> &perTokenScale,
    const int64_t splitItem, const int64_t groupType, const int64_t groupListType, const int64_t actType,
    const c10::optional<c10::IntArrayRef> &tuningConfigOptional)
{
    // 1. 检查所有输入都在 NPU 上，false代表不支持空tensor
    checkTensorOnNPU(x, "x", false);
    checkTensorOnNPU(weight, "weight", false);
    checkTensorOnNPU(bias, "bias", true);
    checkTensorOnNPU(scale, "scale", true);
    checkTensorOnNPU(offset, "offset", true);
    checkTensorOnNPU(antiquantScale, "antiquantScale", true);
    checkTensorOnNPU(antiquantOffset, "antiquantOffset", true);
    checkTensorOnNPU(perTokenScale, "perTokenScale", true);
    checkTensorOnNPU(groupList, "groupList", true);

    // 2. 获取缓存的组合（第一次调用时初始化，后续直接使用）
    const auto &SUPPORTED_COMBOS = getSupportedCombos();

    // 3. 查找匹配的类型组合
    int matched_index = TypeComboManager::findMatchingCombo(SUPPORTED_COMBOS, x, weight, bias, scale, offset,
                                                            antiquantScale, antiquantOffset, groupList, perTokenScale);


    // 4. 如果没找到匹配的组合，生成详细的错误信息
    if (matched_index == -1) {
        // std::stringstream error_msg;
        // error_msg << "Unsupported type combination for groupedmatmul_npu.\n";
        // error_msg << "Got types: x=" << x.scalar_type() << ", bias=" << bias.scalar_type()
        //           << ", scale=" << scale.scalar_type() << ", offset=" << offset.scalar_type()
        //           << ", antiquantScale=" << antiquantScale.scalar_type()
        //           << ", antiquantOffset=" << antiquantOffset.scalar_type() << ", groupList=" <<
        //           groupList.scalar_type()
        //           << ", perTokenScale=" << perTokenScale.scalar_type() << ", weight=" << weight.scalar_type() <<
        //           "\n\n";
        // error_msg << "Supported combinations:\n";

        // for (size_t i = 0; i < SUPPORTED_COMBOS.size(); ++i) {
        //     const auto &combo = SUPPORTED_COMBOS[i];
        //     error_msg << "  Index " << i << ": x=" << combo.x << ", bias=" << combo.bias << ", scale=" << combo.scale
        //               << ", offset=" << combo.offset << ", antiquantScale=" << combo.antiquantScale
        //               << ", antiquantOffset=" << combo.antiquantOffset << ", groupList=" << combo.groupList
        //               << ", perTokenScale=" << combo.perTokenScale << ", weight=" << combo.weight
        //               << ", output=" << combo.output << "\n";
        // }
        // TORCH_CHECK(false, error_msg.str());
        TORCH_CHECK(false, "no match dtype combo");
    }

    // 5. 根据匹配的索引创建输出张量
    const auto &matched_combo = SUPPORTED_COMBOS[matched_index];
    const c10::ScalarType output_type = matched_combo.output;
    // todo 输出shape需要计算
    at::Tensor y = at::empty_like(x[0], at::dtype(output_type));

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    // 6. 使用类型转换模板调用对应的API实现
    // auto acl_call = [=, &matched_combo]() -> int {
    //     // 根据匹配的组合转换所有类型
    //     using XType = typename ScalarTypeToCppType<matched_combo.x>::type;
    //     using BiasType = typename ScalarTypeToCppType<matched_combo.bias>::type;
    //     using ScaleType = typename ScalarTypeToCppType<matched_combo.scale>::type;
    //     using OffsetType = typename ScalarTypeToCppType<matched_combo.offset>::type;
    //     using AntiQuantScaleType = typename ScalarTypeToCppType<matched_combo.antiquantScale>::type;
    //     using AntiQuantOffsetType = typename ScalarTypeToCppType<matched_combo.antiquantOffset>::type;
    //     using GroupListType = typename ScalarTypeToCppType<matched_combo.groupList>::type;
    //     using PerTokenScaleType = typename ScalarTypeToCppType<matched_combo.perTokenScale>::type;
    //     using WeightType = typename ScalarTypeToCppType<matched_combo.weight>::type;
    //     using OutputType = typename ScalarTypeToCppType<matched_combo.output>::type;

    //     // 调用实际的API实现（需要在其他地方定义）
    //     groupedmatmul_api<XType, BiasType, ScaleType, OffsetType, AntiQuantScaleType, AntiQuantOffsetType,
    //                       GroupListType, PerTokenScaleType, WeightType, OutputType>(
    //         stream, x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, y,
    //         splitItem, groupType, groupListType, actType, tuningConfigOptional);
    //     return 0;
    // };
    auto acl_call = [=, &matched_combo]() -> int {
        vector<int64_t> tuning_config_vec; // 临时存储转换后的数据
        vector<int64_t> *tuning_config_ptr = nullptr;

        if (tuningConfigOptional.has_value()) {
            // 将 IntArrayRef 转换为 vector<int64_t>
            tuning_config_vec =
                vector<int64_t>(tuningConfigOptional.value().begin(), tuningConfigOptional.value().end());
            tuning_config_ptr = &tuning_config_vec; // 指向转换后的数据
        }
        dispatch_groupedmatmul(matched_combo, stream, x, weight, bias, scale, offset, antiquantScale, antiquantOffset,
                               groupList, perTokenScale, y, splitItem, groupType, groupListType, actType,
                               tuning_config_ptr);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("GroupedMatmul", acl_call);
    return y;
}

// torch::Tensor groupedmatmul_meta(const torch::Tensor &x, const torch::Tensor &bias, const torch::Tensor &scale,
//                                  const torch::Tensor &offset, const torch::Tensor &antiquantScale,
//                                  const torch::Tensor &antiquantOffset, const torch::Tensor &groupList,
//                                  const torch::Tensor &perTokenScale, const torch::Tensor &weight)
// {
//     TORCH_CHECK(x.defined(), "Input x tensor must be defined");
//     TORCH_CHECK(weight.defined(), "Input weight tensor must be defined");
//     // todo 输出shape需要计算
//     // 2. 获取缓存的组合（第一次调用时初始化，后续直接使用）
//     const auto &SUPPORTED_COMBOS = getSupportedCombos();

//     // 3. 查找匹配的类型组合
//     int matched_index = TypeComboManager::findMatchingCombo(SUPPORTED_COMBOS, x, bias, scale, offset, antiquantScale,
//                                                             antiquantOffset, groupList, perTokenScale, weight);

//     // 4. 如果没找到匹配的组合，生成详细的错误信息
//     if (matched_index == -1) {
//         std::stringstream error_msg;
//         error_msg << "Unsupported type combination for groupedmatmul_npu.\n";
//         error_msg << "Got types: x=" << x.scalar_type() << ", bias=" << bias.scalar_type()
//                   << ", scale=" << scale.scalar_type() << ", offset=" << offset.scalar_type()
//                   << ", antiquantScale=" << antiquantScale.scalar_type()
//                   << ", antiquantOffset=" << antiquantOffset.scalar_type() << ", groupList=" <<
//                   groupList.scalar_type()
//                   << ", perTokenScale=" << perTokenScale.scalar_type() << ", weight=" << weight.scalar_type() <<
//                   "\n\n";
//         error_msg << "Supported combinations:\n";

//         for (size_t i = 0; i < SUPPORTED_COMBOS.size(); ++i) {
//             const auto &combo = SUPPORTED_COMBOS[i];
//             error_msg << "  Index " << i << ": x=" << combo.x << ", bias=" << combo.bias << ", scale=" << combo.scale
//                       << ", offset=" << combo.offset << ", antiquantScale=" << combo.antiquantScale
//                       << ", antiquantOffset=" << combo.antiquantOffset << ", groupList=" << combo.groupList
//                       << ", perTokenScale=" << combo.perTokenScale << ", weight=" << combo.weight
//                       << ", output=" << combo.output << "\n";
//         }

//         TORCH_CHECK(false, error_msg.str());
//     }

//     // 5. 根据匹配的索引创建输出张量
//     const auto &matched_combo = SUPPORTED_COMBOS[matched_index];
//     return torch::empty(x.sizes(), torch::TensorOptions()
//                                        .dtype(ScalarTypeToCppType<matched_combo.output>::type)
//                                        .device(torch::kMeta)
//                                        .memory_format(x.suggest_memory_format()));
// }

// Register Ascend implementations for groupedmatmul
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("groupedmatmul", groupedmatmul_npu);
}

// Register Meta Function for groupedmatmul
// TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
// {
//     m.impl("groupedmatmul", TORCH_FN(groupedmatmul_meta));
// }

} // namespace GroupedMatmul
} // namespace ascend_ops
