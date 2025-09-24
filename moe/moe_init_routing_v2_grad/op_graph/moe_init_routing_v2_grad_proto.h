/* *
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
 * \file moe_init_routing_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOE_INIT_ROUTING_OPS_H_
#define OPS_OP_PROTO_INC_MOE_INIT_ROUTING_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief compute init routing grad for moe input.
 * @par Inputs:
 * @li grad_expanded_x: A 2D or 3D Tensor, tokens squences. Type is:BFloat16, Float16 or Float32. Format support ND.
 * @li expanded_row_idx: A 1D Tensor, token indices in grad_expanded_x. Type is:Int32. Format support ND.
 * @par Outputs:
 * @li grad_x: A 2D Tensor, reverse gradient result. Type is:BFloat16, Float16 or Float32，which is same as that of
 *             grad_expanded_x. axis 0 of grad_x should be same as the value that axis 0 of expanded_row_idx divide
 *             top_k, axis 1 of grad_x should be same as -1 axis of grad_expanded_x. Format support ND.
 * @par Attributes:
 * @li top_k: Required parameter. Type is:Int32. The value must be greater than 0 and can be exactly divided by axis 0
 *            of expanded_row_idx.
 * @li drop_pad_mode: Optional parameter, identify the dropless or drop/pad scenario. Type is:Int32. The value is
 *                    0 (dropless scenario) or 1 (drop/pad scenario).
 * @li active_num: Optional parameter, identify activate scenario. Type is:Int32. The value 0 indicates a non-active
 *                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the size
 *                 of axis 0 of grad_expanded_x must be equal to the value of active_num.
 */
REG_OP(MoeInitRoutingV2Grad)
    .INPUT(grad_expanded_x, "T1")
    .INPUT(expanded_row_idx, "T2")
    .OUTPUT(grad_x, "T1")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .REQUIRED_ATTR(top_k, Int)
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(active_num, Int, 0)
    .OP_END_FACTORY_REG(MoeInitRoutingV2Grad)

} // namespace ge

#endif