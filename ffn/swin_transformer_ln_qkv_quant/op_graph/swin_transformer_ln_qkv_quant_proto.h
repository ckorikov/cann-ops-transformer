/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file fusion_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief
   swin_transformer model specific structure.Operator only supports swin_transformer. \n
* @par Inputs:
* Eight inputs, including:
* @li x: A Tensor. Must be one of the following types: float16.
* @li gamma: A Tensor. Must be one of the following types: float16.
* @li beta: A Tensor. Must be one of the following types: float16.
* @li weight: A Tensor. Must be one of the following types: int8.
* @li bias: A Tensor. Must be one of the following types: float16.
* @li quant_scale: A Tensor. Must be one of the following types: float16.
* @li quant_offset: A Tensor. Must be one of the following types: float16.
* @li dequant_scale: A Tensor. Must be one of the following types: uint64. \n

* @par Attributes:
* @li head_num: A required attribute, the type is int. Defaults to 1.
* @li seq_length: A required attribute, the type is int. Defaults to 32.
* @li epsilon: A required attribute, the type is float. Defaults to 0.000001.
* @li ori_height: A required attribute, the type is int. Defaults to 7
* @li ori_weight: A required attribute, the type is int. Defaults to 7. \n
* @li h_win_szie: A required attribute, the type is int. Defaults to 7. \n
* @li w_win_size: A required attribute, the type is int. Defaults to 7. \n
* @li weight_transpose: A required attribute, the type is bool. Defaults to true. \n

* @par Outputs:
* Three outputs, including:
* @li query_output: A Tensor. Must be one of the following types: float16.
* @li key_output: A Tensor. Must be one of the following types: float16.
* @li value_output: A Tensor. Must be one of the following types: float16. \n
*/
REG_OP(SwinTransformerLnQkvQuant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT16}))
    .INPUT(quant_scale, TensorType({DT_FLOAT16}))
    .INPUT(quant_offset, TensorType({DT_FLOAT16}))
    .INPUT(dequant_scale, TensorType({DT_UINT64}))
    .OUTPUT(query_output, TensorType({DT_FLOAT16}))
    .OUTPUT(key_output, TensorType({DT_FLOAT16}))
    .OUTPUT(value_output, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(seq_length, Int)
    .REQUIRED_ATTR(epsilon, Float)
    .REQUIRED_ATTR(ori_height, Int)
    .REQUIRED_ATTR(ori_weight, Int)
    .REQUIRED_ATTR(h_win_szie, Int)
    .REQUIRED_ATTR(w_win_size, Int)
    .REQUIRED_ATTR(weight_transpose, Bool)
    .OP_END_FACTORY_REG(SwinTransformerLnQkvQuant)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
