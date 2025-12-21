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

#include "onnx_common.h"

namespace domi {

static Status ParseParamsTfIdfVectorizer(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (node == nullptr) {
    std::string reportErrorCode = "E50058";
    std::vector<const char*> errKeys = {"op_name", "description"};
    std::vector<const char*> errValues = {"TfIdfVectorizer", "Dynamic cast op_src to NodeProto failed!"};
    ge::ReportPredefinedErrMsg(reportErrorCode.c_str(), errKeys, errValues);
    return FAILED;
  }

  // 辅助函数：处理数组类型属性
  auto handleArrayAttr = [&](const auto& attr, const std::string& name) {
    if (attr.type() == ge::onnx::AttributeProto::INTS) {
      op_dest.SetAttr(name.c_str(), std::vector<int64_t>{attr.ints().begin(), attr.ints().end()});
    } else if (attr.type() == ge::onnx::AttributeProto::FLOATS) {
      op_dest.SetAttr(name.c_str(), std::vector<float>{attr.floats().begin(), attr.floats().end()});
    } else if (attr.type() == ge::onnx::AttributeProto::STRINGS) {
      op_dest.SetAttr(name.c_str(), std::vector<std::string>{attr.strings().begin(), attr.strings().end()});
    }
  };

  for (const auto& attr : node->attribute()) {
    const std::string& name = attr.name();
    if (attr.type() == ge::onnx::AttributeProto::INT) {
      if (name == "max_gram_length" || name == "max_skip_count" || name == "min_gram_length") {
        op_dest.SetAttr(name.c_str(), attr.i());
      }
    } else if (attr.type() == ge::onnx::AttributeProto::STRING) {
      if (name == "mode") op_dest.SetAttr(name.c_str(), attr.s());
    } else if (name == "weights") {
      handleArrayAttr(attr, name);
    } else if (name == "ngram_counts" || name == "ngram_indexes" || name == "pool_int64s") {
      if (attr.type() == ge::onnx::AttributeProto::INTS) {
        op_dest.SetAttr(name.c_str(), std::vector<int64_t>{attr.ints().begin(), attr.ints().end()});
      }
    } else if (name == "pool_strings") {
      if (attr.type() == ge::onnx::AttributeProto::STRINGS) {
        op_dest.SetAttr(name.c_str(), std::vector<std::string>{attr.strings().begin(), attr.strings().end()});
      }
    }
  }

  return SUCCESS;
}

// register op info to GE
REGISTER_CUSTOM_OP("TfIdfVectorizer")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::9::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::10::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::11::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::12::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::13::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::14::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::15::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::16::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::17::TfIdfVectorizer"),
                   ge::AscendString("ai.onnx::18::TfIdfVectorizer")})
    .ParseParamsFn(ParseParamsTfIdfVectorizer)
    .ImplyType(ImplyType::TVM);
}  // namespace domi