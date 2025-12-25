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
 * \file grouped_matmul_torch.h
 * \brief
 */

// groupedmatmul_npu.cpp
#include <ATen/ATen.h>
// #include <torch_npu/npu_functions.h>
// #include <torch_npu/npu_interface.h>
#include <c10/util/Half.h>
#include <tuple>
#include <vector>
#include <sstream>
#include <type_traits>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"

inline std::string build_error_msg(const std::string& func_name, const std::string& name, 
                                  const std::string& message) {
    std::ostringstream oss;
    oss << func_name << name << message;
    return oss.str();
}

// ScalarType 到 C++ 类型的转换模板（需在头文件中定义）
template <c10::ScalarType ScalarType>
struct ScalarTypeToCppType;

template <>
struct ScalarTypeToCppType<at::kHalf> {
    using type = c10::Half;
};

template <>
struct ScalarTypeToCppType<at::kBFloat16> {
    using type = at::BFloat16;
};

template <>
struct ScalarTypeToCppType<at::kFloat> {
    using type = float;
};

template <>
struct ScalarTypeToCppType<at::kDouble> {
    using type = double;
};

template <>
struct ScalarTypeToCppType<at::kInt> {
    using type = int32_t;
};

template <>
struct ScalarTypeToCppType<at::kLong> {
    using type = int64_t;
};

// 类型组合结构
struct TypeCombo {
    c10::ScalarType x;
    c10::ScalarType bias;
    c10::ScalarType scale;
    c10::ScalarType offset;
    c10::ScalarType antiquantScale;
    c10::ScalarType antiquantOffset;
    c10::ScalarType groupList;
    c10::ScalarType perTokenScale;
    c10::ScalarType weight;
    c10::ScalarType output;
};

// ===================== 第一步：定义类型 ↔ ID 映射 =====================
// 1. 枚举所有需要支持的 at::ScalarType，并分配唯一 ID
enum class ScalarTypeId : int {
    HALF = 0,          // at::kHalf → 0
    BFLOAT16 = 1,      // at::kBFloat16 → 1
    CHAR = 2,          // at::kChar → 2 (对应 int8_t)
    FLOAT = 3,         // at::kFloat → 3
    INT = 4,           // at::kInt → 4 (对应 int32_t)
    // 按需补充其他类型（如 at::kUInt8 → 5 等）
};

// 2. at::ScalarType → ScalarTypeId（整型 ID）的映射表
const std::unordered_map<at::ScalarType, ScalarTypeId> kScalarTypeToId = {
    {at::kHalf, ScalarTypeId::HALF},
    {at::kBFloat16, ScalarTypeId::BFLOAT16},
    {at::kChar, ScalarTypeId::CHAR},
    {at::kFloat, ScalarTypeId::FLOAT},
    {at::kInt, ScalarTypeId::INT},
    // 补充其他支持的类型
};

// 3. ScalarTypeId → 具体 C++ 类型的模板映射（核心：编译期类型绑定）
template <ScalarTypeId Id>
struct IdToCppType;

// 特化：每个 ID 绑定对应的 C++ 类型
template <> struct IdToCppType<ScalarTypeId::HALF> { using type = at::Half; };
template <> struct IdToCppType<ScalarTypeId::BFLOAT16> { using type = at::BFloat16; };
template <> struct IdToCppType<ScalarTypeId::CHAR> { using type = int8_t; }; // at::kChar 对应 int8_t
template <> struct IdToCppType<ScalarTypeId::FLOAT> { using type = float; };
template <> struct IdToCppType<ScalarTypeId::INT> { using type = int32_t; };
// 补充其他类型的特化

// 辅助别名：简化类型获取
template <ScalarTypeId Id>
using IdToCppType_t = typename IdToCppType<Id>::type;

// 4. 辅助函数：安全获取类型 ID（带错误检查）
ScalarTypeId getScalarTypeId(at::ScalarType scalar_type) {
    auto it = kScalarTypeToId.find(scalar_type);
    TORCH_CHECK(it != kScalarTypeToId.end(), 
                "Unsupported ScalarType: ", scalar_type, " (no mapped ID)");
    return it->second;
}


// 类型组合管理器
class TypeComboManager {
public:
    static std::vector<TypeCombo> createCombosFromLists(
        const std::vector<c10::ScalarType> &x_list, const std::vector<c10::ScalarType> &bias_list,
        const std::vector<c10::ScalarType> &scale_list, const std::vector<c10::ScalarType> &offset_list,
        const std::vector<c10::ScalarType> &antiquantScale_list,
        const std::vector<c10::ScalarType> &antiquantOffset_list, const std::vector<c10::ScalarType> &groupList_list,
        const std::vector<c10::ScalarType> &perTokenScale_list, const std::vector<c10::ScalarType> &weight_list,
        const std::vector<c10::ScalarType> &output_list)
    {
        std::vector<TypeCombo> combos;
        combos.reserve(x_list.size());

        for (size_t i = 0; i < x_list.size(); ++i) {
            combos.push_back({x_list[i], bias_list[i], scale_list[i], offset_list[i], antiquantScale_list[i],
                              antiquantOffset_list[i], groupList_list[i], perTokenScale_list[i], weight_list[i],
                              output_list[i]});
        }

        return combos;
    }


    template <typename TensorContainer>
    static bool checkTensorType(const TensorContainer &container, c10::ScalarType expected_type, const std::string& name = "curTensor", bool allow_empty = true)
    {
        const std::string& func_name="[checkTensorType]";

        // 处理 optional 类型
        if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::TensorList>> ||
                      std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
            if (!container.has_value()) {
                TORCH_CHECK(allow_empty,build_error_msg(func_name, name, " is not provided (null optional), but empty is not allowed"));
                return allow_empty;
            }
        }
        // 获取实际的值或引用
        auto get_value = [&]() -> auto &
        {
            if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::TensorList>>) {
                return container.value();
            } else if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
                return container.value();
            } else if constexpr (std::is_same_v<TensorContainer, torch::TensorList>) {
                return container;
            } else if constexpr (std::is_same_v<TensorContainer, torch::Tensor>) {
                return container;
            }
        };

        const auto &value = get_value();

        // 处理 Tensor 和 TensorList 的不同检查逻辑
        if constexpr (std::is_same_v<TensorContainer, torch::Tensor> ||
                      std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
            // 单个 Tensor 的处理
            if (!value.defined()) {
                TORCH_CHECK(allow_empty,build_error_msg(func_name, name,  " tensor is not defined"));
                return allow_empty;
            }
            if (value.scalar_type() != expected_type) {
                return false;
            }
            return true;
        } else {
            // TensorList 的处理
            if (value.empty()) {
                TORCH_CHECK(allow_empty, build_error_msg(func_name, name, " tensor list is empty, but empty is not allowed") );
                return allow_empty;
            }

            for (size_t i = 0; i < value.size(); ++i) {
                const torch::Tensor &tensor = value[i];
                if (!tensor.defined()) {
                    TORCH_CHECK(allow_empty, build_error_msg(func_name, name, " tensor list contains undefined tensor") );
                    return allow_empty;
                }
                if (tensor.scalar_type() != expected_type) {
                    return false;
                }
            }
            return true;
        }
    }


    static int findMatchingCombo(const std::vector<TypeCombo> &combos, const torch::TensorList &x,
                                 const torch::TensorList &weight, const c10::optional<torch::TensorList> &bias,
                                 const c10::optional<torch::TensorList> &scale,
                                 const c10::optional<torch::TensorList> &offset,
                                 const c10::optional<torch::TensorList> &antiquantScale,
                                 const c10::optional<torch::TensorList> &antiquantOffset,
                                 const c10::optional<torch::Tensor> &groupList,
                                 const c10::optional<torch::TensorList> &perTokenScale)
    {
        for (size_t i = 0; i < combos.size(); ++i) {
            const auto &combo = combos[i];
            if (checkTensorType(x, combo.x, "x", false) && checkTensorType(weight, combo.weight, "weight", false) &&
                checkTensorType(bias, combo.bias, "bias", true) && checkTensorType(scale, combo.scale, "scale", true) &&
                checkTensorType(offset, combo.offset, "offset", true) &&
                checkTensorType(antiquantScale, combo.antiquantScale, "antiquantScale", true) &&
                checkTensorType(antiquantOffset, combo.antiquantOffset, "antiquantOffset", true) &&
                checkTensorType(perTokenScale, combo.perTokenScale, "perTokenScale", true) &&
                checkTensorType(groupList, combo.groupList, "groupList", true)) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }
};

template <typename TensorContainer>
void checkTensorOnNPU(const TensorContainer &container, const std::string& name = "curTensor", bool allow_empty = false)
{
    const std::string& func_name ="[checkTensorOnNPU]";

    // 处理 optional 类型
    if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::TensorList>> ||
                  std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
        if (!container.has_value()) {
            TORCH_CHECK(allow_empty, build_error_msg(func_name, name,  " is not provided (null optional), but empty is not allowed"));
            return;
        }
    }
    // 获取实际的值或引用
    auto get_value = [&]() -> auto &
    {
        if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::TensorList>>) {
            return container.value();
        } else if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
            return container.value();
        } else if constexpr (std::is_same_v<TensorContainer, torch::TensorList>) {
            return container;
        } else if constexpr (std::is_same_v<TensorContainer, torch::Tensor>) {
            return container;
        }
    };
    const auto &value = get_value();
    // 处理 Tensor 和 TensorList 的不同检查逻辑
    if constexpr (std::is_same_v<TensorContainer, torch::Tensor> ||
                  std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
        // 单个 Tensor 的处理
        if (!value.defined()) {
            TORCH_CHECK(allow_empty, build_error_msg(func_name, name,  " tensor is undefined, but empty is not allowed") );
            return;
        }

        if (!torch_npu::utils::is_npu(value)) {
            TORCH_CHECK(false, build_error_msg(func_name, name,  " tensor must be on NPU device"));
        }
    } else {
        // TensorList 的处理
        if (value.empty()) {
            TORCH_CHECK(allow_empty, build_error_msg(func_name, name,  " tensor list is empty, but empty is not allowed"));
            return;
        }
        for (size_t i = 0; i < value.size(); ++i) {
            const torch::Tensor &tensor = value[i];
            if (tensor.defined() && !torch_npu::utils::is_npu(tensor)) {
                std::string msg = " tensor at index " + std::to_string(i) + " must be on NPU device";
                TORCH_CHECK(false, build_error_msg(func_name, name,msg) );
            }
        }
    }
}

template<typename TensorType, typename ElementType>
ElementType* get_first_tensor_address(const TensorType& input, bool allow_empty = false) {
    const auto& get_tensor = [&]() -> const torch::Tensor* {
        // 处理 optional<torch::Tensor>
        if constexpr (std::is_same_v<TensorType, c10::optional<torch::Tensor>>) {
            if (!input.has_value()) {
                if (!allow_empty) TORCH_CHECK(false, "optional<Tensor> has no value");
                return nullptr;
            }
            if (!input->defined()) {
                if (!allow_empty) TORCH_CHECK(false, "optional<Tensor> is undefined");
                return nullptr;
            }
            return &input.value();
        }
        // 处理 optional<TensorList>
        else if constexpr (std::is_same_v<TensorType, c10::optional<at::TensorList>>) {
            if (!input.has_value()) {
                if (!allow_empty) TORCH_CHECK(false, "optional<TensorList> has no value");
                return nullptr;
            }
            if (input->empty()) {
                if (!allow_empty) TORCH_CHECK(false, "optional<TensorList> is empty");
                return nullptr;
            }
            const auto& tensor = input.value()[0];
            if (!tensor.defined()) {
                if (!allow_empty) TORCH_CHECK(false, "First tensor in optional<TensorList> is undefined");
                return nullptr;
            }
            return &tensor;
        }
        // 处理 TensorList
        else if constexpr (std::is_same_v<TensorType, at::TensorList>) {
            if (input.empty()) {
                if (!allow_empty) TORCH_CHECK(false, "TensorList is empty");
                return nullptr;
            }
            const auto& tensor = input[0];
            if (!tensor.defined()) {
                if (!allow_empty) TORCH_CHECK(false, "First tensor in TensorList is undefined");
                return nullptr;
            }
            return &tensor;
        }
        // 处理 torch::Tensor
        else if constexpr (std::is_same_v<TensorType, torch::Tensor>) {
            if (!input.defined()) {
                if (!allow_empty) TORCH_CHECK(false, "Tensor is undefined");
                return nullptr;
            }
            return &input;
        }
        // 不支持的类型
        else {
            static_assert(std::is_same_v<TensorType, void>, 
                         "Unsupported tensor type");
            return nullptr;
        }
    };
    
    const torch::Tensor* tensor_ptr = get_tensor();
    return tensor_ptr ? tensor_ptr->data_ptr<ElementType>() : nullptr;
}