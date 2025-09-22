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
 * \file apply_rotary_pos_emb_def.cpp
 * \brief
 */
#include <vector>
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> qDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
static const std::vector<ge::Format> qFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ApplyRotaryPosEmb : public OpDef {
public:
    explicit ApplyRotaryPosEmb(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType(qDtype)
            .Format(qFormat)
            .UnknownShapeFormat(qFormat)
            .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType(qDtype)
            .Format(qFormat)
            .UnknownShapeFormat(qFormat)
            .AutoContiguous();
        this->Input("cos")
            .ParamType(REQUIRED)
            .DataType(qDtype)
            .Format(qFormat)
            .UnknownShapeFormat(qFormat)
            .AutoContiguous();
        this->Input("sin")
            .ParamType(REQUIRED)
            .DataType(qDtype)
            .Format(qFormat)
            .UnknownShapeFormat(qFormat)
            .AutoContiguous();
        this->Output("query").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).UnknownShapeFormat(qFormat);
        this->Output("key").ParamType(REQUIRED).DataType(qDtype).Format(qFormat).UnknownShapeFormat(qFormat);
        this->Attr("layout").AttrType(OPTIONAL).Int(1);
        this->Attr("rotary_mode").AttrType(OPTIONAL).String("half");
        OpAICoreConfig configApply;
        configApply.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "apply_rotary_pos_emb");
        this->AICore().AddConfig("ascend910b", configApply);
        this->AICore().AddConfig("ascend910_93", configApply);
        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "apply_rotary_pos_emb_apt");
        this->AICore().AddConfig("ascend910_95", regbaseCfg);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        aicore_config.Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Input("cos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Input("sin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Output("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Output("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.ExtendCfgInfo("opFile.value", "apply_rotary_pos_emb");
        this->AICore().AddConfig("ascend310p", aicore_config);
        this->AICore().AddConfig("ascend910", aicore_config);
    }
};

OP_ADD(ApplyRotaryPosEmb);
} // namespace ops
