/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_INFERSHAPE_WEIGHT_QUANT_CHECKER_H_
#define GROUPED_MATMUL_INFERSHAPE_WEIGHT_QUANT_CHECKER_H_

#include "graph/utils/type_utils.h"
#include "grouped_matmul_infershape_common_util.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

class GroupedMatmulWeightQuantChecker {
public:
    GroupedMatmulWeightQuantChecker(){};
    ~GroupedMatmulWeightQuantChecker(){};
    graphStatus GetXAndWeightDimValue(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs);
    graphStatus CheckShape(const gert::InferShapeContext *context, const GroupedMatmulCommonUtil &commonUtil);
    graphStatus InferOutShape(gert::InferShapeContext *context) const;
    graphStatus CheckDtype(const gert::InferDataTypeContext *context) const;
    graphStatus InferOutDtype(gert::InferDataTypeContext *context) const;

private:
    graphStatus CheckShapeForXAndWeight(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;
    graphStatus CheckShapeForTensorList(const gert::InferShapeContext *context, size_t gmm_index,
                                        const std::string &tensorType) const;
    graphStatus CheckFormatValid(const gert::InferShapeContext *context) const;
    graphStatus CheckScenarioValidForShape(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;
    graphStatus CheckShapeValid(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs);
    graphStatus CheckShapeForWeightQuantParam(const gert::InferShapeContext *context) const;
    graphStatus CheckShapeForGrouplist(const gert::InferShapeContext *context, const gert::Shape *groupListShape) const;
    graphStatus UpdateShapeY(gert::InferShapeContext *context, size_t idxY, std::vector<int64_t> &yDims) const;
    graphStatus CheckGroupSize(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;
    graphStatus CheckTransposeValid(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;

private:
    int64_t groupNum_; //当前含义为M分组数g
    int64_t xKDim_;
    int64_t xMDim_;
    int64_t weightKDim_;
    int64_t weightNDim_;
    size_t xdimNum_;
    size_t weightdimNum_;
};

}  // namespace ops

#endif  // GROUPED_MATMUL_INFERSHAPE_DAVID_QUANT_CHECKER_H_