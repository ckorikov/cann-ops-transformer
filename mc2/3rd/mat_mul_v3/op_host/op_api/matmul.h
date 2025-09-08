/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* MatMulNd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor);

const aclTensor* MatMulNdFp162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor);

const aclTensor* MatMulNz(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor);
/*
包括ND输入NZ输出和NZ输入NZ输出两种切K模式
*/
const aclTensor* MatMulNzFp162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor);

// 输入self=ND，输入mat2=NZ
const aclTensor* MatMulNdNz(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor);

// 输入self=NZ，输入mat2=NZ, 输出ND
const aclTensor* MatMulNzNzNd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor);

const aclTensor* MatMulV3Nd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool transposeX1, const bool transposeX2,
    const bool offsetX, const bool enableHf32, aclOpExecutor* executor);

// 输入ND
const aclTensor* GemmV3Nd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* c, bool transposeX1, bool transposeX2, bool enableHf32,
    aclOpExecutor* executor);

} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_OP_H_
