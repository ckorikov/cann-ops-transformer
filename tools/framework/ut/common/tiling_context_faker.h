/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_DEV_TESTS_UT_COMMON_TILING_CONTEXT_FAKER_H
#define OPS_MATH_DEV_TESTS_UT_COMMON_TILING_CONTEXT_FAKER_H

#include "op_tiling_context_builder.h"

namespace gert {

class TilingContextPara {
public:
    class TensorDescription {
    public:
        TensorDescription(const gert::StorageShape& shape, ge::DataType dtype, ge::Format format) :
            shape_(shape), dtype_(dtype), format_(format) {}
    public:
        gert::StorageShape shape_;
        ge::DataType dtype_ = ge::DT_FLOAT;
        ge::Format format_ = ge::FORMAT_ND;
    };
public:
    TilingContextPara(const std::string& opName,
                      const std::vector<TensorDescription>& inputTensorDesc,
                      const std::vector<TensorDescription>& outputTensorDesc,
                      void* compileInfo = nullptr,
                      uint64_t coreNum = 64,
                      uint64_t ubSize = 262144,
                      uint64_t tilingDataSize = 4096) : 
                      opName_(opName),
                      inputTensorDesc_(inputTensorDesc),
                      outputTensorDesc_(outputTensorDesc),
                      compileInfo_(compileInfo),
                      coreNum_(coreNum),
                      ubSize_(ubSize),
                      tilingDataSize_(tilingDataSize) {}

public:
    std::string opName_;
    std::vector<TensorDescription> inputTensorDesc_;
    std::vector<TensorDescription> outputTensorDesc_;
    uint64_t coreNum_        = 64;
    uint64_t ubSize_         = 262144;
    uint64_t tilingDataSize_ = 4096;
    void* compileInfo_ = nullptr;
};

class TilingContextFaker : public OpTilingContextBuilder {
public:
    TilingContextFaker& SetOpType(const std::string opType);

    /* only one can be choosed from IrInstanceNum */
    TilingContextFaker& NodeIoNum(size_t inputNum, size_t outputNum);

    /* can be used for dynamic inputs/outputs
     * only one can be choosed from NodeIoNum */
    TilingContextFaker& IrInstanceNum(const std::vector<uint32_t>& inputInstanceNum,
                                      const std::vector<uint32_t>& outputInstanceNum);

    TilingContextFaker& NodeInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                    ge::Format storageFormat);

    TilingContextFaker& NodeOutputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                     ge::Format storageFormat);

    template<typename T, typename = typename std::enable_if<
        std::disjunction<
            std::is_same<T, bool>,
            std::is_same<T, int64_t>,
            std::is_same<T, float>,
            std::is_same<T, const AscendString &>,
            std::is_same<T, const std::vector<bool> &>,
            std::is_same<T, const std::vector<int64_t> &>,
            std::is_same<T, const std::vector<float> &>,
            std::is_same<T, const std::vector<AscendString> &>,
            std::is_same<T, const std::vector<std::vector<int64_t>> &>
        >::value
    >::type>
    TilingContextFaker& Attr(const std::string& attrName, T attr) {
        OpTilingContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }

    TilingContextFaker& InputTensors(const std::vector<Tensor *>& inputTensors);

    TilingContextFaker& OutputTensors(const std::vector<Tensor *>& outputTensors);

    TilingContextFaker& CompileInfo(const void* compileInfo);

    TilingContextFaker& PlatformInfo(const void* platformInfo);

    TilingContextFaker& DeterministicInfo(int32_t* deterministicInfo);

    TilingContextFaker& TilingData(const void* tilingData);

    TilingContextFaker& Workspace(const ContinuousVector* workspace);

    ContextHolder<TilingContext> Build();
};

} // namespace gert
#endif // OPS_MATH_DEV_TESTS_UT_COMMON_INFERSHAPE_CONTEXT_FAKER_H
