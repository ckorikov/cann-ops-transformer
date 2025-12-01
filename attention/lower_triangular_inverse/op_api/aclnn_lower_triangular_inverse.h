/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_CHUNK_GATED_DELTA_RULE_INVERSE_H
#define OP_API_INC_CHUNK_GATED_DELTA_RULE_INVERSE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLowerTriangularInverseGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：实现下三角求逆
 * @param [in] x: matmul左矩阵，数据类型支持：float32。
 * @param [out] y: 计算结果，数据类型：float32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnLowerTriangularInverseGetWorkspaceSize(const aclTensor *x, aclTensor *y, 
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnLowerTriangularInverse
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口 aclnnLowerTriangularInverseGetWorkspaceSize 获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnLowerTriangularInverse(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CHUNK_GATED_DELTA_RULE_INVERSE_H