/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_quant_metadata_proto.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_METADATA_PROTO_H
#define LIGHTNING_INDEXER_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Function LightningIndexerMetadata.

* @par Inputs:
* @li actual_seq_lengths_query: A matrix tensor. The type support int32.
* Effective sequence length of q in different batches.
* @li actual_seq_lengths_key: A matrix tensor. The type support int32.
* Effective sequence length of key/value in different batches.

* @par Attributes:
* @li aic_core_num: An int. cube core num of device
* @li aiv_core_num: An int. vector core num of device
* @li layout_query
* @li layout_key

* @par Outputs:
* @li metadata: A matrix tensor. The type support int32.
* The output of attention structure.
*/
REG_OP(LightningIndexerQuantMetadata)
    .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_key, TensorType({DT_INT32}))
    .OUTPUT(metadata, TensorType({DT_INT32}))
    .REQUIRED_ATTR(aic_core_num, Int)
    .REQUIRED_ATTR(aiv_core_num, Int)
    .REQUIRED_ATTR(batch_size, Int)
    .REQUIRED_ATTR(query_seq_size, Int)
    .REQUIRED_ATTR(query_head_num, Int)
    .REQUIRED_ATTR(kv_seq_size, Int)
    .REQUIRED_ATTR(kv_head_num, Int)
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_key, String, "PA_BSND")
    .ATTR(sparse_mode, Int, 3)
    .ATTR(soc_version, String, "ascend910B")
    .ATTR(is_fd, Bool, false)
    .ATTR(pre_tokens, Int, 0)
    .ATTR(next_tokens, Int, 0)
    .ATTR(cmp_ratio, Int, 4)
    .OP_END_FACTORY_REG(LightningIndexerQuantMetadata)
} // namespace ge

#endif // LIGHTNING_INDEXER_METADATA_PROTO_H

// FD