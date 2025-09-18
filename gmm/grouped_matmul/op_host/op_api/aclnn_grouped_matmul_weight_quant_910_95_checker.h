/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_MATMUL_WEIGHT_QUANT_910_95_CHECKER_H
#define OP_API_INC_GROUPED_MATMUL_WEIGHT_QUANT_910_95_CHECKER_H
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_grouped_matmul_util.h"

namespace gmm {
class AclnnGroupedMatmulWeightQuant91095Checker {
public:
    explicit AclnnGroupedMatmulWeightQuant91095Checker(const GroupedMatmulParams &gmmParams) : gmmParams_(gmmParams){};
    ~AclnnGroupedMatmulWeightQuant91095Checker(){};
    aclnnStatus CheckGroupedMatmulWeightQuant91095();

private:
    aclnnStatus CheckGmmQuantParamsEmpty() const;
    aclnnStatus CheckTensorListDtype(const aclTensorList *tensorList, const DataType &dtype,
                                     const DataType &weightDtype) const;
    aclnnStatus CheckTensorListShape(const aclTensorList *tensorList, const std::string &tensorType) const;

    aclnnStatus CheckWeightFormatAndShape(const DataType &weightDtype) const;
    aclnnStatus CheckTransposeStatus(const DataType &weightDtype) const;
    aclnnStatus CheckNKValue() const;

    aclnnStatus CheckBiasDtype(const DataType &xDtype) const;
    aclnnStatus CheckAntiQuantDtype(const DataType &xDtype, const DataType &weightDtype) const;
    aclnnStatus CheckAntiQuantShape() const;
    aclnnStatus CheckUnsupportApi(const DataType &weightDtype) const;
    aclnnStatus CheckGroupSize() const;

private:
    GroupedMatmulParams gmmParams_;
};
}  // namespace gmm
#endif