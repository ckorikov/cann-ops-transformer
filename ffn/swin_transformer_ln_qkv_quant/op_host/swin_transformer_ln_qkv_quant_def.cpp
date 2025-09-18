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
 * \file swin_transformer_ln_qkv_quant.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> xDtype = {
    ge::DT_FLOAT16
};
static const std::vector<ge::Format> xFormat = {
    ge::FORMAT_ND
};
class SwinTransformerLnQkvQuant : public OpDef {
public:
    explicit SwinTransformerLnQkvQuant(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
	        .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
	        .AutoContiguous();
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
	        .AutoContiguous();
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
	        .AutoContiguous();
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
	        .AutoContiguous();
        this->Input("quant_scale")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
	        .AutoContiguous();
        this->Input("quant_offset")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
	        .AutoContiguous();
        this->Input("dequant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64})
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
	        .AutoContiguous();
        this->Output("query_output")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat);
        this->Output("key_output")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat);
        this->Output("value_output")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat);
        this->Attr("head_num").Int();
        this->Attr("seq_length").Int();
        this->Attr("epsilon").Float(0.000001); // epsilon default 0.000001
        this->Attr("ori_height").Int();
        this->Attr("ori_weight").Int();
        this->Attr("h_win_size").Int();
        this->Attr("w_win_size").Int();
        this->Attr("weight_transpose").Bool(true);  // true is B transposed

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
              .DynamicFormatFlag(true)
              .DynamicRankSupportFlag(true)
              .DynamicShapeSupportFlag(true)
              .NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend310p", config);
    }
};

OP_ADD(SwinTransformerLnQkvQuant);
}