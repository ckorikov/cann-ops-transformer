/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_grouped_matmul_add_case.cpp
 * \brief GroupedMatmulAdd Aclnn 测试用例.
 */

#include <utility>
#include "tests/utils/log.h"
#include "aclnn_grouped_matmul_add.h"
#include "aclnn_grouped_matmul_add_v2.h"
#include "aclnn_grouped_matmul_add_case.h"

using namespace ops::adv::tests::grouped_matmul_add;

bool GroupedMatmulAddTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnGroupedMatmulAddCase *>(curCase);
    auto *aclnnParam = &cs->mAclnnParam;

    aclnnStatus ret = ACL_SUCCESS;
    if (aclnnParam->mAclnnGroupedMatmulAddVersion == AclnnGroupedMatmulAddVersion::V1) {
        ret = aclnnGroupedMatmulAddGetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensor(), aclnnParam->aclnnWeight.GetAclTensor(),
            aclnnParam->aclnnGroupList.GetAclTensor(), aclnnParam->aclnnY.GetAclTensor(),
            aclnnParam->mTransposeX, aclnnParam->mTransposeWeight,
            aclnnParam->mGroupType, workSpaceSize, opExecutor);
    } else if (aclnnParam->mAclnnGroupedMatmulAddVersion == AclnnGroupedMatmulAddVersion::V2) {
        ret = aclnnGroupedMatmulAddV2GetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensor(), aclnnParam->aclnnWeight.GetAclTensor(),
            aclnnParam->aclnnGroupList.GetAclTensor(), aclnnParam->aclnnY.GetAclTensor(),
            aclnnParam->mTransposeX, aclnnParam->mTransposeWeight,
            aclnnParam->mGroupType, aclnnParam->mGroupListType,
            workSpaceSize, opExecutor);
    }
    return ret == ACL_SUCCESS;
}

bool GroupedMatmulAddKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnGroupedMatmulCase *>(curCase);
    auto *aclnnParam = &cs->mAclnnParam;
    auto *aclnnCtx = &cs->mAclnnCtx;

    aclnnStatus ret = ACL_SUCCESS;
    if (aclnnParam->mAclnnGroupedMatmulAddVersion == AclnnGroupedMatmulAddVersion::V1) {
        ret = aclnnGroupedMatmul(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(),
                                 aclnnCtx->GetAclOpExecutor(), aclnnCtx->GetAclRtStream());
    } else if (aclnnParam->mAclnnGroupedMatmulAddVersion == AclnnGroupedMatmulAddVersion::V2) {
        ret = aclnnGroupedMatmulV2(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(),
                                   aclnnCtx->GetAclOpExecutor(), aclnnCtx->GetAclRtStream());
    }
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnGroupedMatmulAdd failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}

AclnnGroupedMatmulAddCase::AclnnGroupedMatmulAddCase()
    : GroupedMatmulAddCase(), mAclnnCtx(AclnnContext()), mAclnnParam(AclnnGroupedMatmulAddParam())
{
}

AclnnGroupedMatmulAddCase::AclnnGroupedMatmulAddCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo,
                                               AclnnGroupedMatmulParam aclnnParam, int32_t tilingTemplatePriority)
    : GroupedMatmulCase(name, enable, dbgInfo, std::move(opInfo), Param(), tilingTemplatePriority),
      mAclnnParam(std::move(aclnnParam))
{
}

bool AclnnGroupedMatmulAddCase::InitParam()
{
    return mAclnnParam.Init();
}

bool AclnnGroupedMatmulAddCase::InitOpInfo()
{
    if (!GroupedMatmulCase::InitOpInfo()) {
        return false;
    }

    auto rst = mAclnnCtx.SetOpName(this->mOpInfo.mName.c_str());
    rst = rst && mAclnnCtx.SetTilingRunCbf(GroupedMatmulAddTilingRunCbf);
    rst = rst && mAclnnCtx.SetKernelRunCbf(GroupedMatmulAddKernelRunCbf);
    rst = rst && mOpInfo.SetContext(&mAclnnCtx);
    return rst;
}

bool AclnnGroupedMatmulAddCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool AclnnGroupedMatmulAddCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}
