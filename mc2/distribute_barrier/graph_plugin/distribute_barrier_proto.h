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
* @brief DistributeBarrier operator interface implementation.

* @par Inputs
* One inputs, including:
* @li x_ref: An optional tensor, reserved. Support dtype:bfloat16, float16, float32, bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64. Support format: ND.

* @par Attributes
* @li group: Required. Input comm group name, means experts parallelism, dtype: String.
* @li world_size: Required. Input comm world size, dtype: int64.

* @par Outputs
* One outputs, including:
* @li x_ref: A tensor. reserved. Support dtype:bfloat16, float16, float32, bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64. Support format: ND.
*/
REG_OP(DistributeBarrier)
    .INPUT(x_ref, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(x_ref, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .OP_END_FACTORY_REG(DistributeBarrier)

}  // namespace ge


#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
