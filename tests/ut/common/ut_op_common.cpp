/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ut_op_common.cpp
 */

#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "ut_op_util.h"

static const std::map<std::string, ge::AnyValue::ValueType> kAttrTypesMap = {
  {"VT_INT", ge::AnyValue::ValueType::VT_INT},
  {"VT_BOOL", ge::AnyValue::ValueType::VT_BOOL},
  {"VT_FLOAT", ge::AnyValue::ValueType::VT_FLOAT},
  {"VT_STRING", ge::AnyValue::ValueType::VT_STRING},
  {"VT_LIST_INT", ge::AnyValue::ValueType::VT_LIST_INT},
  {"VT_LIST_BOOL", ge::AnyValue::ValueType::VT_LIST_BOOL},
  {"VT_LIST_FLOAT", ge::AnyValue::ValueType::VT_LIST_FLOAT},
  {"VT_LIST_LIST_INT", ge::AnyValue::ValueType::VT_LIST_LIST_INT},
};

static uint8_t* GetConstTensor(ge::Operator& op, const size_t index, int64_t shape_size) {
  ge::Tensor const_tensor;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_name = op_desc->GetInputNameByIndex(index);
  uint8_t* data = nullptr;
  size_t size = 0;
  if (op.GetInputConstData(input_name.c_str(), const_tensor) == ge::GRAPH_SUCCESS) {
    size = const_tensor.GetSize();
    data = const_tensor.GetData();
  }
  ge::DataType const_dtype = op_desc->MutableInputDesc(index)->GetDataType();
  uint8_t* input_tensor_holder = new uint8_t[sizeof(gert::Tensor) + size];
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder);
  int64_t value_size = shape_size;
  if (size > 0) {
    std::memcpy(input_tensor + 1, data, size);
    value_size =
        (const_dtype == ge::DT_INT64 || const_dtype == ge::DT_UINT64) ? size / sizeof(int64_t) : size / sizeof(int32_t);
  }
  gert::Tensor tensor({{value_size}, {value_size}},        // shape
                      {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                      (size > 0) ? gert::kFollowing : gert::kOnHost,   // placement
                      const_dtype,                         // dt
                      nullptr);
  std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
  return input_tensor_holder;
}

ge::graphStatus InferShapeTest(ge::Operator& op, const Runtime2TestParam& param) {
  ge::graphStatus ret;
  size_t input_size = op.GetInputsSize();
  std::vector<std::string> attrs = param.attrs;
  std::vector<bool> input_const = param.input_const;
  std::vector<uint32_t> irnum = param.irnum;
  if (irnum.size() > 0) {
    if (input_const.size() == 0) input_const.assign(irnum.size(), false);
  } else if (input_const.size() > 0) {
    if (irnum.size() == 0) irnum.assign(input_const.size(), 1);
  } else {
    input_const.assign(input_size, false);
    irnum.assign(input_size, 1);
  }
  std::string optype = op.GetOpType();
  size_t output_size = op.GetOutputsSize();
  auto faker = gert::InferShapeContextFaker()
                    .SetOpType(optype)
                    .NodeIoNum(input_size, output_size)
                    .IrInstanceNum(irnum);

  vector<uint8_t*> const_tensors;
  std::vector<gert::StorageShape> input_shapes(input_size);
  std::vector<void *> input_shapes_ref(input_size);
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);
  if (operator_info == nullptr) return GRAPH_FAILED;
  if (input_size > 0) {
    size_t count = 0;
    for (size_t i = 0; i < input_const.size(); ++i) {
      if (input_const[i]) {
        auto input_desc = operator_info->MutableInputDesc(i);
        uint8_t* input_tensor_holder = GetConstTensor(op, i, input_desc->MutableShape().GetShapeSize());
        if (input_tensor_holder == nullptr) return GRAPH_FAILED;
        input_shapes_ref[count] = input_tensor_holder;
        const_tensors.push_back(input_tensor_holder);
        ge::Format input_format = input_desc->GetFormat();
        ge::Format origin_format = input_desc->GetOriginFormat();
        ge::DataType dtype = input_desc->GetDataType();
        faker = faker.NodeInputTd(count, dtype, origin_format, input_format);
        count++;
      } else for (int idx = 0; idx < irnum[i]; idx++) {
        size_t idx_off = i + idx;
        if (i > 0) {
          auto irnum_i = irnum[i-1] == 0 ? 1 : irnum[i-1];
          idx_off = irnum_i * i + idx;
        }
        auto input_desc = operator_info->MutableInputDesc(idx_off);
        if (input_desc == nullptr) continue;
        ge::Format input_format = input_desc->GetFormat();
        ge::Format origin_format = input_desc->GetOriginFormat();
        ge::DataType dtype = input_desc->GetDataType();
        faker = faker.NodeInputTd(count, dtype, origin_format, input_format);
        for (int64_t dim : input_desc->GetOriginShape().GetDims()) {
          input_shapes[count].MutableOriginShape().AppendDim(dim);
        }
        for (int64_t dim : input_desc->MutableShape().GetDims()) {
          input_shapes[count].MutableStorageShape().AppendDim(dim);
        }
        input_shapes_ref[count] = &input_shapes[count];
        count++;
      }
    }
    faker = faker.InputShapes(input_shapes_ref);
  }

  std::vector<gert::StorageShape> output_shapes(output_size);
  std::vector<void *> output_shapes_ref(output_size);
  if (output_size > 0) {
    ge::TensorDesc tensor_desc = create_desc({-2});
    for (size_t i = 0; i < output_size; ++i) {
      output_shapes_ref[i] = &output_shapes[i];
      op.UpdateOutputDesc(operator_info->GetOutputNameByIndex(i), tensor_desc);
    }
    faker = faker.OutputShapes(output_shapes_ref);
  }

  auto op_attrs_map = op.GetAllAttrNamesAndTypes();
  std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value;
  std::pair<std::string, ge::AnyValue> p;
  if (attrs.size() > 0) {
    for (auto item : attrs) {
      p.first = item;
      auto attr_it = op_attrs_map.find(item);
      if (attr_it != op_attrs_map.end()) {
        auto type_it = kAttrTypesMap.find(attr_it->second);
        if (type_it != kAttrTypesMap.end()) {
          switch (type_it->second) {
            case ge::AnyValue::ValueType::VT_BOOL: {
              bool value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<bool>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_INT: {
              int64_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<int64_t>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_FLOAT: {
              float32_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<float32_t>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_STRING: {
              std::string value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::string>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_INT: {
              std::vector<int64_t> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::vector<int64_t>>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_FLOAT: {
              std::vector<float32_t> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::vector<float32_t>>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_BOOL: {
              std::vector<bool> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::vector<bool>>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_LIST_INT: {
              std::vector<std::vector<int64_t>> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(value);
            }
            break;
            default:
              std::cout << "[ERROR]"<<__FILE__<<":"<<__LINE__<<"The ValueType is not supported!" << std::endl;
          }
        }
      }
      keys_to_value.push_back(p);
    }
    faker = faker.NodeAttrs(keys_to_value);
  }

  auto holder = faker.Build();
  if (gert::OpImplRegistry::GetInstance().GetOpImpl(optype.c_str()) == nullptr) return GRAPH_FAILED;
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype.c_str())->infer_shape;
  if (infer_shape_func == nullptr) return GRAPH_FAILED;
  gert::InferShapeContext *context = holder.GetContext<gert::InferShapeContext>();
  if (context == nullptr) return GRAPH_FAILED;
  ret = infer_shape_func(context);
  for (uint8_t* tensor : const_tensors) { delete []tensor; }
  for (size_t i = 0; i < output_size; i++) {
    auto out_shape = context->GetOutputShape(i);
    if (out_shape == nullptr) return GRAPH_FAILED;
    auto output_desc = operator_info->MutableOutputDesc(i);
    output_desc->SetShape(GeShape(ut_util::ToVector(*out_shape)));
  }
  return ret;
}

ge::graphStatus InferDataTypeTest(ge::Operator& op, const Runtime2TestParam& param) {
  ge::graphStatus ret;
  size_t input_size = op.GetInputsSize();
  std::vector<std::string> attrs = param.attrs;
  std::vector<bool> input_const = param.input_const;
  std::vector<uint32_t> irnum = param.irnum;
  if (irnum.size() > 0) {
    if (input_const.size() == 0) input_const.assign(irnum.size(), false);
  } else if (input_const.size() > 0) {
    if (irnum.size() == 0) irnum.assign(input_const.size(), 1);
  } else {
    input_const.assign(input_size, false);
    irnum.assign(input_size, 1);
  }
  std::string optype = op.GetOpType();
  size_t output_size = op.GetOutputsSize();
  auto faker = gert::InferDataTypeContextFaker()
                    .NodeIoNum(input_size, output_size)
                    .IrInstanceNum(irnum);

  vector<uint8_t*> const_tensors;
  std::vector<ge::DataType> input_datatype(input_size);
  std::vector<void *> input_datatype_ref(input_size);
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);
  if (operator_info == nullptr) return GRAPH_FAILED;
  if (input_size > 0) {
    size_t count = 0;
    for (size_t i = 0; i < input_const.size(); ++i) {
      if (input_const[i]) {
        auto input_desc = operator_info->MutableInputDesc(i);
        uint8_t* input_tensor_holder = GetConstTensor(op, i, input_desc->MutableShape().GetShapeSize());
        if (input_tensor_holder == nullptr) return GRAPH_FAILED;
        input_datatype_ref[count] = input_tensor_holder;
        const_tensors.push_back(input_tensor_holder);
        ge::Format input_format = input_desc->GetFormat();
        ge::Format origin_format = input_desc->GetOriginFormat();
        ge::DataType dtype = input_desc->GetDataType();
        faker = faker.NodeInputTd(count, dtype, origin_format, input_format);
        count++;
      } else for (int idx = 0; idx < irnum[i]; idx++) {
        auto input_desc = operator_info->MutableInputDesc(i + idx);
        if (input_desc == nullptr) continue;
        ge::Format input_format = input_desc->GetFormat();
        ge::Format origin_format = input_desc->GetOriginFormat();
        ge::DataType dtype = input_desc->GetDataType();
        faker = faker.NodeInputTd(count, dtype, origin_format, input_format);
        input_datatype[count] = dtype;
        input_datatype_ref[count] = &input_datatype[count];
        count++;
      }
    }
    faker = faker.InputDataTypes(input_datatype_ref);
  }

  std::vector<ge::DataType> output_datatype(output_size);
  std::vector<void *> output_datatype_ref(output_size);
  if (output_size > 0) {
    ge::TensorDesc tensor_desc = create_desc({-2});
    for (size_t i = 0; i < output_size; ++i) {
      output_datatype_ref[i] = &output_datatype[i];
      op.UpdateOutputDesc(operator_info->GetOutputNameByIndex(i), tensor_desc);
    }
    faker = faker.OutputDataTypes(output_datatype_ref);
  }

  auto op_attrs_map = op.GetAllAttrNamesAndTypes();
  std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value;
  std::pair<std::string, ge::AnyValue> p;
  if (attrs.size() > 0) {
    for (auto item : attrs) {
      p.first = item;
      auto attr_it = op_attrs_map.find(item);
      if (attr_it != op_attrs_map.end()) {
        auto type_it = kAttrTypesMap.find(attr_it->second);
        if (type_it != kAttrTypesMap.end()) {
          switch (type_it->second) {
            case ge::AnyValue::ValueType::VT_BOOL: {
              bool value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<bool>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_INT: {
              int64_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<int64_t>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_FLOAT: {
              float32_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<float32_t>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_STRING: {
              std::string value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::string>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_INT: {
              std::vector<int64_t> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::vector<int64_t>>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_FLOAT: {
              std::vector<float32_t> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::vector<float32_t>>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_BOOL: {
              std::vector<bool> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::vector<bool>>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_LIST_INT: {
              std::vector<std::vector<int64_t>> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(value);
            }
            break;
            default:
              std::cout << "[ERROR]"<<__FILE__<<":"<<__LINE__<<"The ValueType is not supported!" << std::endl;
          }
        }
      }
      keys_to_value.push_back(p);
    }
    faker = faker.NodeAttrs(keys_to_value);
  }

  auto holder = faker.Build();
  if (gert::OpImplRegistry::GetInstance().GetOpImpl(optype.c_str()) == nullptr) return GRAPH_FAILED;
  auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype.c_str())->infer_datatype;
  if (infer_datatype_func == nullptr) return GRAPH_FAILED;
  gert::InferDataTypeContext *context = holder.GetContext<gert::InferDataTypeContext>();
  if (context == nullptr) return GRAPH_FAILED;
  ret = infer_datatype_func(context);
  for (uint8_t* tensor : const_tensors) { delete []tensor; }
  for (size_t i = 0; i < output_size; i++) {
    auto output_datatype = context->GetOutputDataType(i);
    auto output_desc = operator_info->MutableOutputDesc(i);
    output_desc->SetDataType(output_datatype);
  }
  return ret;
}

ge::graphStatus InferShapeTest(ge::Operator& op) {
  Runtime2TestParam param;
  return InferShapeTest(op, param);
}

ge::graphStatus InferDataTypeTest(ge::Operator& op) {
  Runtime2TestParam param;
  return InferDataTypeTest(op, param);
}
