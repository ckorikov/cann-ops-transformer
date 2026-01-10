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
 * \file l0_lightning_indexer_quant_metadata.cpp
 * \brief
 */

#include "l0_lightning_indexer_quant_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(LightningIndexerQuantMetadata);

const aclTensor* LightningIndexerQuantMetadata(
    const aclTensor* actualSeqLengthsQueryOptional,
    const aclTensor* actualSeqLengthsKeyOptional,
    int64_t aicCoreNum,
    int64_t aivCoreNum,
    int64_t batchSize,
    int64_t querySeqSize,
    int64_t queryHeadNum,
    int64_t kvSeqSize,
    int64_t kvHeadNum,
    char* layoutQueryOptional,
    char* layoutKeyOptional,
    int64_t sparseModeOptional,
    char* socVersionOptional,
    bool isFdOptional,
    int64_t  preTokensOptional,
    int64_t  nextTokensOptional,
    int64_t cmpRatioOptional,
    const aclTensor* metaData,
    aclOpExecutor* executor) {
  L0_DFX(LightningIndexerQuantMetadata, actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, aicCoreNum, 
         aivCoreNum, batchSize, querySeqSize, queryHeadNum, kvSeqSize, kvHeadNum, layoutQueryOptional, 
         layoutKeyOptional, sparseModeOptional, socVersionOptional, isFdOptional, preTokensOptional, nextTokensOptional, cmpRatioOptional, metaData);

  static internal::AicpuTaskSpace space("LightningIndexerQuantMetadata");

  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
      LightningIndexerQuantMetadata,
      OP_ATTR_NAMES({"aic_core_num", "aiv_core_num", "batch_size", "query_seq_size",
                     "query_head_num", "kv_seq_size", "kv_head_num", "layout_query",
                     "layout_key", "sparse_mode", "soc_version", "is_fd","pre_tokens","next_tokens","cmp_ratio"}),
      OP_INPUT(actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional), OP_OUTPUT(metaData),
      OP_ATTR(aicCoreNum, aivCoreNum, batchSize, querySeqSize, queryHeadNum, kvSeqSize, kvHeadNum,
              layoutQueryOptional, layoutKeyOptional, sparseModeOptional, socVersionOptional, isFdOptional, preTokensOptional, nextTokensOptional, cmpRatioOptional));
  OP_CHECK(ret == ACL_SUCCESS,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                   "LightningIndexerQuantMetadata"
                   " ADD_TO_LAUNCHER_LIST_AICPU failed."),
           return nullptr);
  return metaData;
}
}  // namespace l0op
