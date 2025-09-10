/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef NN_TESTS_UT_COMMON_KERNEL_RUN_CONTEXT_FACKER_H_
#define NN_TESTS_UT_COMMON_KERNEL_RUN_CONTEXT_FACKER_H_

#include <memory>
#include <vector>
#include <cstring>
#include "runtime/kernel_run_context.h"
#include "runtime/context_extend.h"
#include "runtime/storage_shape.h"
#include "runtime/tiling_context.h"
#include "lowering/buffer_pool.h"
#include "any_value.h"
#include "node.h"

namespace gert {
struct KernelRunContextHolder {
  template<typename T>
  T *GetContext() {
    return reinterpret_cast<T*>(context);
  }
  ComputeNodeInfo *MutableComputeNodeInfo() {
    return reinterpret_cast<ComputeNodeInfo *>(compute_node_extend_holder.get());
  }
  size_t kernel_input_num;
  size_t kernel_output_num;
  std::unique_ptr<uint8_t[]> context_holder;
  std::vector<AsyncAnyValue> value_holder;
  std::unique_ptr<uint8_t[]> compute_node_extend_holder;
  bg::BufferPool buffer_pool;
  KernelRunContext *context;
};
KernelRunContextHolder BuildKernelRunContext(size_t input_num, size_t output_num);

class KernelRunContextFaker {
 public:
  KernelRunContextFaker() = default;
  KernelRunContextFaker &SetOpType(const std::string op_type);
  KernelRunContextFaker &KernelIONum(size_t input_num, size_t output_num);
  KernelRunContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  KernelRunContextFaker &IrInputNum(size_t input_num);
  KernelRunContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num);
  KernelRunContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                     ge::Format storage_format);
  KernelRunContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format);
  KernelRunContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value);
  KernelRunContextFaker &Inputs(std::vector<void *> inputs);
  KernelRunContextFaker &Outputs(std::vector<void *> outputs);

  KernelRunContextHolder Build();

 private:
  ge::NodePtr FakeNode();

 private:
  size_t kernel_input_num_;
  size_t kernel_output_num_;
  size_t node_input_num_;
  size_t node_output_num_;
  std::string optype_ = "node";
  std::vector<uint32_t> ir_instance_num_;
  std::vector<CompileTimeTensorDesc> node_input_tds_;
  std::vector<CompileTimeTensorDesc> node_output_tds_;
  std::vector<void *> inputs_;
  std::vector<void *> outputs_;
  std::vector<std::pair<std::string, ge::AnyValue>> attrs_;
  ge::ComputeGraphPtr graph_;
};

class InferShapeContextFaker {
 public:
  InferShapeContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  InferShapeContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  InferShapeContextFaker &SetOpType(std::string optype) {
    base_faker_.SetOpType(optype);
    return *this;
  }
  InferShapeContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  InferShapeContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                       ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }

  InferShapeContextFaker &InputShapes(std::vector<void *> input_shapes);
  InferShapeContextFaker &OutputShapes(std::vector<void *> output_shapes);

  KernelRunContextHolder Build();

 private:
  enum InputsAppend { kInputsInferShapeFunc, kInputsAppendEnd };

 private:
  KernelRunContextFaker base_faker_;
};

class ExeResGenerationContextFaker {
 public:
  ExeResGenerationContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  ExeResGenerationContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  ExeResGenerationContextFaker &SetOpType(std::string optype) {
    base_faker_.SetOpType(optype);
    return *this;
  }
  ExeResGenerationContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  ExeResGenerationContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  ExeResGenerationContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                       ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  ExeResGenerationContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }

  ExeResGenerationContextFaker &InputShapes(std::vector<void *> input_shapes);
  ExeResGenerationContextFaker &OutputShapes(std::vector<void *> output_shapes);

  KernelRunContextHolder Build();

 private:
  enum InputsAppend { kInputsGenTaskFunc, kInputsAppendEnd };

 private:
  std::vector<void *> inputs_;
  std::vector<void *> outputs_;
  KernelRunContextFaker base_faker_;
};

class InferShapeRangeContextFaker {
 public:
  InferShapeRangeContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  InferShapeRangeContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  InferShapeRangeContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  InferShapeRangeContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeRangeContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                       ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferShapeRangeContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }

  InferShapeRangeContextFaker &InputShapeRanges(std::vector<void *> input_shape_ranges);
  InferShapeRangeContextFaker &OutputShapeRanges(std::vector<void *> output_shape_ranges);

  KernelRunContextHolder Build();

 private:
  enum InputsAppend { kInputsInferShapeRangeFunc, kInputsAppendEnd };

 private:
  KernelRunContextFaker base_faker_;
};

class InferDataTypeContextFaker {
 public:
  InferDataTypeContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  InferDataTypeContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  InferDataTypeContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  InferDataTypeContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferDataTypeContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                       ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  InferDataTypeContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }

  InferDataTypeContextFaker &InputDataTypes(std::vector<void *> input_datatypes);
  InferDataTypeContextFaker &OutputDataTypes(std::vector<void *> output_datatypes);

  KernelRunContextHolder Build();

 private:
  enum InputsAppend { kInputsInferDataTypeFunc, kInputsAppendEnd };

 private:
  std::vector<void *> inputs_;
  std::vector<void *> outputs_;
  KernelRunContextFaker base_faker_;
};

class TilingContextFaker {
 public:
  TilingContextFaker &SetOpType(const std::string op_type) {
    base_faker_.SetOpType(op_type);
    return *this;
  }
  TilingContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  TilingContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  TilingContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  TilingContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format, ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  TilingContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                   ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }
  TilingContextFaker &InputShapes(std::vector<void *> input_shapes);
  TilingContextFaker &OutputShapes(std::vector<void *> output_shapes);
  TilingContextFaker &CompileInfo(void *compile_info);
  TilingContextFaker &PlatformInfo(void *platform_info);
  TilingContextFaker &TilingData(void *tiling_data);
  TilingContextFaker &Workspace(ContinuousVector *workspace);
  TilingContextFaker &ConstInput(std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors);
  TilingContextFaker &DeterministicInfo(void *deterministic_info);

  KernelRunContextHolder Build();

 private:
  void UpdateInputs();

 private:
  enum InputsAppend { kInputsCompileInfo, kInputsPlatformInfo, kInputsTilingFunc, kInputsDeterministic, kInputsAppendEnd };

  KernelRunContextFaker base_faker_;
  std::vector<void *> input_shapes_;
  std::vector<void *> output_shapes_;
  std::vector<void *> outputs_ {TilingContext::kOutputNum};

  void *platform_info_;
  void *compile_info_;
  void *deterministic_info_;
};
}  // namespace gert
#endif  //NN_TESTS_UT_COMMON_KERNEL_RUN_CONTEXT_FACKER_H_
