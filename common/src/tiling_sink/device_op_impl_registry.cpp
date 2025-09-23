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
 * \file device_op_impl_registry.cpp
 * \brief
 */
#include "register/device_op_impl_registry.h"
#include <memory>
#include "device_op_impl_registry_impl.h"
#include "err/ops_err.h"

namespace optiling {
DeviceOpImplRegistry &DeviceOpImplRegistry::GetSingleton()
{
  static DeviceOpImplRegistry g_deviceOpImplRegistry;
  return g_deviceOpImplRegistry;
}

void DeviceOpImplRegistry::RegisterSinkTiling(std::string &opType, SinkTilingFunc &func)
{
  std::string opTypeString = opType;
  sinkTilingFuncsMap_[opTypeString] = func;
}

SinkTilingFunc DeviceOpImplRegistry::GetSinkTilingFunc(std::string &opType)
{
  std::string opTypeString = opType;
  auto func = sinkTilingFuncsMap_.find(opTypeString);
  if (func == sinkTilingFuncsMap_.end()) {
    return nullptr;
  }
  return func->second;
}

std::string& DeviceOpImplRegisterImpl::GetOpType()
{
  return opType_;
}

DeviceOpImplRegisterImpl::~DeviceOpImplRegisterImpl()
{
}

DeviceOpImplRegister::DeviceOpImplRegister(const char *opType)
{
  impl_ = std::make_unique<DeviceOpImplRegisterImpl>();
  impl_->GetOpType() = opType;
}

DeviceOpImplRegister &DeviceOpImplRegister::Tiling(SinkTilingFunc func)
{
  DeviceOpImplRegistry::GetSingleton().RegisterSinkTiling(impl_->GetOpType(), func);
  return *this;
}

DeviceOpImplRegister::DeviceOpImplRegister(DeviceOpImplRegister &&other) noexcept
{
  impl_ = std::make_unique<DeviceOpImplRegisterImpl>();
  impl_->GetOpType() = other.impl_->GetOpType();
}

DeviceOpImplRegister::DeviceOpImplRegister(const DeviceOpImplRegister &other)
{
  impl_ = std::make_unique<DeviceOpImplRegisterImpl>();
  impl_->GetOpType() = other.impl_->GetOpType();
}

DeviceOpImplRegister::~DeviceOpImplRegister() 
{
}
}