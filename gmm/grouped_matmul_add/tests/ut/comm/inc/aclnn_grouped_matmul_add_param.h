/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_grouped_matmul_add_param.h
 * \brief GroupedMatmulAdd Aclnn 参数信息.
 */

#ifndef UTEST_ACLNN_GROUPED_MATMUL_ADD_PARAM_H
#define UTEST_ACLNN_GROUPED_MATMUL_ADD_PARAM_H

#include "grouped_matmul_add_case.h"
#include "tests/utils/aclnn_tensor.h"
#include "tests/utils/aclnn_tensor_list.h"

namespace ops::adv::tests::grouped_matmul_add {

class AclnnGroupedMatmulAddParam : public ops::adv::tests::grouped_matmul_add::Param {
public:
    using AclnnTensor = ops::adv::tests::utils::AclnnTensor;
    using AclnnTensorList = ops::adv::tests::utils::AclnnTensorList;

public:
    enum class AclnnGroupedMatmulAddVersion {
        V1,
        V2,
};

public:
    AclnnGroupedMatmulAddVersion mAclnnGroupedMatmulVersion = AclnnGroupedMatmulAddVersion::V1;
    /* 输入输出 */
    AclnnTensor aclnnX, aclnnWeight, aclnnGroupList,aclnnY;
    std::map<std::string, Tensor> mTensors;

public:
    AclnnGroupedMatmulAddParam() = default;
    AclnnGroupedMatmulAddParam(std::vector<Tensor> inputs, std::vector<int64_t> groupListData,
                            bool transposeX, bool transposeWeight, int32_t groupType, int32_t groupListType,
                            AclnnGroupedMatmulAddVersion aclnnGroupedMatmulAddVersion);

    ~AclnnGroupedMatmulAddParam();

    bool Init();

private:
    bool InitGroupList();
};

} // namespace ops::adv::tests::grouped_matmul
#endif // UTEST_ACLNN_GROUPEDMATMUL_ADD_PARAM_H
