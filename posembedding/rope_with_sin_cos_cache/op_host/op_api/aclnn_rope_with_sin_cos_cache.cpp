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
 * \file aclnn_rope_with_sin_cos_cache.cpp
 * \brief
 */

#include "rope_with_sin_cos_cache.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "add.h"
#include "rsqrt.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_rope_with_sin_cos_cache.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/op_api_def.h"

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "iostream"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t QUERY_OUT_INDEX = 0;
static const int64_t KEY_OUT_INDEX = 1;

static const int64_t DIM_ONE = 0;
static const int64_t DIM_TWO = 1;

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16,  op::DataType::DT_FLOAT,
    op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> emptyDtypes = {};

static const std::initializer_list<DataType>& GetSupportDtypeList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return emptyDtypes;
    }
}

static bool CheckNotNull(
    const aclTensor* positions, const aclTensor* queryIn, const aclTensor* keyIn, const aclTensor* cosSinCache,
    const aclTensor* queryOut, const aclTensor* keyOut)
{
    OP_CHECK_NULL(positions, return false);
    OP_CHECK_NULL(queryIn, return false);
    OP_CHECK_NULL(keyIn, return false);
    OP_CHECK_NULL(cosSinCache, return false);
    OP_CHECK_NULL(queryOut, return false);
    OP_CHECK_NULL(keyOut, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* positions, const aclTensor* queryIn, const aclTensor* keyIn, const aclTensor* cosSinCache,
    const aclTensor* queryOut, const aclTensor* keyOut)
{
    // 检查positions queryIn的数据类型是否在支持列表内
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    const auto& supportList = GetSupportDtypeList();
    if (supportList.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", ToString(socVersion).GetString());
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(positions, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(queryIn, supportList, return false);

    // keyIn, cosSinCache, queryOut, keyOut的数据类型是否与out一致
    OP_CHECK_DTYPE_NOT_MATCH(keyIn, queryIn->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(cosSinCache, queryIn->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(queryOut, queryIn->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(keyOut, queryIn->GetDataType(), return false);
    return true;
}

static int64_t GetTensorNumel(const aclTensor* x, size_t startIdx)
{
    size_t xShapeDim = x->GetViewShape().GetDimNum();
    int64_t xShapeSize = 1;
    for (size_t i = startIdx; i < xShapeDim; i++) {
        xShapeSize *= x->GetViewShape().GetDim(i);
    }
    return xShapeSize;
}

static bool CheckShape(
    const aclTensor* positions, const aclTensor* queryIn, const aclTensor* keyIn, const aclTensor* cosSinCache,
    const aclTensor* queryOut, const aclTensor* keyOut)
{
    // 检查输入的所有shape是不是2维；
    if (queryIn->GetViewShape().GetDimNum() != 2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected queryIn to be a vector of size 2, "
            "but got queryIn of shape %s.",
            op::ToString(queryIn->GetViewShape()).GetString());
        return false;
    }

    if (keyIn->GetViewShape().GetDimNum() != 2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected keyIn to be a vector of size 2, "
            "but got keyIn of shape %s.",
            op::ToString(keyIn->GetViewShape()).GetString());
        return false;
    }

    if (cosSinCache->GetViewShape().GetDimNum() != 2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected cosSinCache to be a vector of size 2, "
            "but got cosSinCache of shape %s.",
            op::ToString(cosSinCache->GetViewShape()).GetString());
        return false;
    }
    // 检查keyIn的1维度是不是等于queryIn的一维
    OP_CHECK(
        keyIn->GetViewShape()[0] == queryIn->GetViewShape()[0],
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected keyIn.shape[0] == queryIn->GetViewShape()[0] to be true, but got false."),
        return false);
    // 检查输入和输出的shape 是否相同
    OP_CHECK_SHAPE_NOT_EQUAL(queryOut, queryIn, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(keyOut, keyIn, return false);

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* positions, const aclTensor* queryIn, const aclTensor* keyIn, const aclTensor* cosSinCache,
    const aclIntArray* mropeSection, int64_t headSize, bool isNeoxStyle, aclTensor* queryOut, aclTensor* keyOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(positions, queryIn, keyIn, cosSinCache, queryOut, keyOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(positions, queryIn, keyIn, cosSinCache, queryOut, keyOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(positions, queryIn, keyIn, cosSinCache, queryOut, keyOut), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRopeWithSinCosCacheGetWorkspaceSize(
    const aclTensor* positions, const aclTensor* queryIn, const aclTensor* keyIn, const aclTensor* cosSinCache,
    const aclIntArray* mropeSection, int64_t headSize, bool isNeoxStyle, aclTensor* queryOut, aclTensor* keyOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnRopeWithSinCosCache, DFX_IN(positions, queryIn, keyIn, cosSinCache, mropeSection, headSize, isNeoxStyle),
        DFX_OUT(queryOut, keyOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入cosSinCosCache转换成连续的tensor
    auto cosSinCacheContiguous = l0op::Contiguous(cosSinCache, uniqueExecutor.get());
    CHECK_RET(cosSinCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，参数检查
    auto ret =
        CheckParams(positions, queryIn, keyIn, cosSinCache, mropeSection, headSize, isNeoxStyle, queryOut, keyOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(headSize != 0, ACLNN_ERR_PARAM_INVALID);
    int64_t numQheads = queryIn->GetViewShape()[1] / headSize;
    int64_t numKheads = keyIn->GetViewShape()[1] / headSize;

    int64_t queryStride = queryIn->GetViewStrides()[0];
    int64_t keyStride = keyIn->GetViewStrides()[0];

    std::tuple<aclTensor*, aclTensor*> result;
    result = l0op::RopeWithSinCosCache(
        positions, queryIn, keyIn, cosSinCacheContiguous, mropeSection, headSize, isNeoxStyle, queryStride, keyStride,
        numQheads, numKheads, uniqueExecutor.get());
    auto query = std::get<0>(result);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto key = std::get<1>(result);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto queryViewCopyResult = l0op::ViewCopy(query, queryOut, uniqueExecutor.get());
    CHECK_RET(queryViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyViewCopyResult = l0op::ViewCopy(key, keyOut, uniqueExecutor.get());
    CHECK_RET(keyViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，获取计算过程中需要使用的workspace大小--ok
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// 固定写法，获取计算过程中需要使用的workspace大小--ok
aclnnStatus aclnnRopeWithSinCosCache(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRopeWithSinCosCache);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
