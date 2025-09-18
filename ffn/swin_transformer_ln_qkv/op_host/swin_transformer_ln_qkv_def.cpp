/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file swin_transformer_ln_qkv.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class SwinTransformerLnQKV : public OpDef {
public:
    explicit SwinTransformerLnQKV(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("query_output")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("key_output")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("value_output")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Attr("head_num")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("head_dim")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("seq_length")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("shifts")
            .AttrType(OPTIONAL)
            .ListInt({});
        this->Attr("epsilon")
            .AttrType(OPTIONAL)
            .Float(0.0000001f);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SwinTransformerLnQKV);
}