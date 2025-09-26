/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFERSHAPE_CONTEXT_FAKER_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFERSHAPE_CONTEXT_FAKER_H

#include "op_infer_shape_context_builder.h"

namespace gert {

class InferShapeContextFaker : public OpInferShapeContextBuilder {
public:
    InferShapeContextFaker& SetOpType(const std::string opType);

    /* only one can be choosed from IrInstanceNum */
    InferShapeContextFaker& NodeIoNum(size_t inputNum, size_t outputNum);

    /* can be used for dynamic inputs/outputs
     * only one can be choosed from NodeIoNum */
    InferShapeContextFaker& IrInstanceNum(const std::vector<uint32_t>& inputInstanceNum,
                                          const std::vector<uint32_t>& outputInstanceNum);

    InferShapeContextFaker& NodeInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                        ge::Format storageFormat);

    InferShapeContextFaker& NodeOutputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                         ge::Format storageFormat);

    InferShapeContextFaker& Attr(const std::string& attrName, bool attr) {
        OpInferShapeContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    InferShapeContextFaker& Attr(const std::string& attrName, int64_t attr) {
        OpInferShapeContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    InferShapeContextFaker& Attr(const std::string& attrName, float attr) {
        OpInferShapeContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    InferShapeContextFaker& Attr(const std::string& attrName, const AscendString& attr) {
        OpInferShapeContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    InferShapeContextFaker& Attr(const std::string& attrName, const std::vector<bool>& attr) {
        OpInferShapeContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    InferShapeContextFaker& Attr(const std::string& attrName, const std::vector<int64_t>& attr) {
        OpInferShapeContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    InferShapeContextFaker& Attr(const std::string& attrName, const std::vector<float>& attr) {
        OpInferShapeContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    InferShapeContextFaker& Attr(const std::string& attrName, const std::vector<AscendString>& attr) {
        OpInferShapeContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    InferShapeContextFaker& Attr(const std::string& attrName, const std::vector<std::vector<int64_t>>& attr) {
        OpInferShapeContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }

    InferShapeContextFaker& InputTensors(const std::vector<Tensor *>& inputTensors);

    InferShapeContextFaker& OutputShapes(const std::vector<StorageShape *>& outputShapes);

    ContextHolder<InferShapeContext> Build();
};

} // namespace gert
#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFERSHAPE_CONTEXT_FAKER_H
