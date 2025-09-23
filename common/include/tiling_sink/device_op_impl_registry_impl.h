/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file device_op_impl_registry_impl.h
 * \brief
 */

#ifndef OP_TILING_DEVICE_OP_IMPL_REGISTRY_IMPL_H
#define OP_TILING_DEVICE_OP_IMPL_REGISTRY_IMPL_H

#include <string>
#include <map>
#include "register/device_op_impl_registry.h"

namespace optiling {
class DeviceOpImplRegistry {
 public:
  static DeviceOpImplRegistry& GetSingleton();
  void RegisterSinkTiling(std::string &opType, SinkTilingFunc& func);
  SinkTilingFunc GetSinkTilingFunc(std::string &opType);

 private:
  DeviceOpImplRegistry() = default;
  ~DeviceOpImplRegistry() = default;

 private:
  std::map<std::string, SinkTilingFunc> sinkTilingFuncsMap_;
};

class DeviceOpImplRegisterImpl {
 public:
  DeviceOpImplRegisterImpl() = default;
  ~DeviceOpImplRegisterImpl();
  std::string& GetOpType();

 private:
  std::string opType_ = "";
};
}  // namespace optiling

#endif