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
 * \file moe_compute_expert_tokens_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOECOMPUTEEXPERT_H_
#define OPS_OP_PROTO_INC_MOECOMPUTEEXPERT_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Binary finds the position of the last row processed by each expert in the sorted_experts array.
 * @par Inputs:
 * @li sorted_experts: An 1D Tensor, sorted expert array. Type is:Int32.
 * @par Outputs:
 * @li total_rows_before_expert: A Tensor. Type is:Int32.
 * @par Attributes:
 * @li num_experts: Required parameter. Type is:Int. The value must be more than 0 and less than 2147483647.
 */
REG_OP(MoeComputeExpertTokens)
    .INPUT(sorted_experts, "T")
    .OUTPUT(total_rows_before_expert, "T")
    .REQUIRED_ATTR(num_experts, Int)
    .DATATYPE(T, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(MoeComputeExpertTokens)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MOECOMPUTEEXPERT_H_
