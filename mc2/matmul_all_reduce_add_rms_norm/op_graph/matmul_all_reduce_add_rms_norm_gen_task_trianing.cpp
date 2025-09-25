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
 * \file matmul_all_reduce_gen_task.cc
 * \brief
 */
#include "error_util.h"
#include "error/ops_error.h"
#include "checker.h"
#include "graph/utils/args_format_desc_utils.h"
#include "platform/platform_info.h"
#include "register/op_ct_impl_registry.h"
#include "mc2_gen_task_utils.h"

namespace ops {
inline static bool IsPlatform310P()
{
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    GE_ASSERT_SUCCESS(
        fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info));
    OPS_ERR_IF(
        fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
            ge::GRAPH_SUCCESS,
        OPS_LOG_E("", "Cannot get platform info!"), return false);
    OPS_LOG_D("", "Get soc version: %s", optional_info.soc_version.c_str());
    static const std::vector<std::string> version_with_common_def = {"Ascend310P", "Ascend310P1", "Ascend310P3"};
    return find(version_with_common_def.begin(), version_with_common_def.end(), optional_info.soc_version) !=
           version_with_common_def.end();
}

// 51的MatmulAlleduce已不再演进，这部分代码直接使用老版本代码，仅因入参原型不同做了适配，暂不整改，后续可能会删除
inline static ge::Status InsertHiddenInputForAicore310P(
    const gert::ExeResGenerationContext* context, domi::TaskDef& aicore_task)
{
    auto task_def = aicore_task.mutable_kernel_with_handle();
    auto task_ctx = task_def->mutable_context();
    const std::string args_format = task_ctx->args_format();
    std::vector<ge::ArgDesc> arg_descs;
    OP_CHECK(
        ge::ArgsFormatDescUtils::Parse(args_format, arg_descs) != ge::GRAPH_SUCCESS || arg_descs.empty(),
        OP_LOGE(context->GetNodeName(), "Failed to parse, args_format:[%s]", args_format.c_str()),
        return ge::GRAPH_FAILED);
    size_t insert_idx = 0U;
    for (; insert_idx < arg_descs.size(); ++insert_idx) {
        if (arg_descs[insert_idx].addr_type == ge::AddrType::INPUT) {
            break;
        }
    }

    OP_LOGD(context->GetNodeName(), "Origin format is %s, insertion position is %zu.", args_format.c_str(), insert_idx);
    // set args into task
    arg_descs.insert(
        arg_descs.begin() + insert_idx,
        {ge::AddrType::HIDDEN_INPUT, static_cast<int32_t>(ge::HiddenInputsType::HCOM), false, {0}});
    const std::string new_args_format = ge::ArgsFormatDescUtils::Serialize(arg_descs);
    task_ctx->set_args_format(new_args_format);
    OP_LOGD(context->GetNodeName(), "New format is %s.", new_args_format.c_str());
    const uint32_t args_size = task_def->args_size();
    const uint32_t sing_args_size = 8;
    size_t total_args_size = args_size + sing_args_size;
    void* all_args_buff = (void*)malloc(total_args_size);
    if (all_args_buff == nullptr) {
        return ge::GRAPH_FAILED;
    }
    errno_t sec_ret = memcpy_s(
        all_args_buff, static_cast<size_t>(total_args_size), task_def->args().data(),
        static_cast<size_t>(insert_idx * sing_args_size));
    if (sec_ret != EOK) {
        free(all_args_buff);
        return ge::GRAPH_FAILED;
    }
    uint8_t* cast_ptr = reinterpret_cast<uint8_t*>(all_args_buff);
    errno_t sec_ret1 = memcpy_s(
        cast_ptr + insert_idx * sing_args_size + sing_args_size,
        static_cast<size_t>(args_size - insert_idx * sing_args_size),
        task_def->args().data() + insert_idx * sing_args_size,
        static_cast<size_t>(args_size - insert_idx * sing_args_size));
    if (sec_ret1 != EOK) {
        free(all_args_buff);
        return ge::GRAPH_FAILED;
    }
    task_def->set_args(all_args_buff, total_args_size);
    free(all_args_buff);
    return ge::GRAPH_SUCCESS;
}

inline ge::Status MatmulAllReduceGenTaskCallback(
    const gert::ExeResGenerationContext* context, std::vector<domi::TaskDef>& tasks)
{
    if (IsPlatform310P()) {
        int64_t aicore_idx = Mc2GenTaskUtils::GetTaskIdxByType(context, tasks, RT_MODEL_TASK_KERNEL);
        if (aicore_idx < 0) {
            aicore_idx = Mc2GenTaskUtils::GetTaskIdxByType(context, tasks, RT_MODEL_TASK_ALL_KERNEL);
        }
        OP_CHECK(
            aicore_idx < 0, OPS_LOG_E(context->GetNodeName(), "Failed to get AICore task."), return ge::GRAPH_FAILED);
        OPS_LOG_I(
            context->GetNodeName(), "Start to generate task for MC2, task def size %lu, aicore index %ld.",
            tasks.size(), aicore_idx);
        GE_ASSERT_SUCCESS(InsertHiddenInputForAicore310P(context, tasks[static_cast<size_t>(aicore_idx)]));
        domi::TaskDef aicpu_task{};
        GE_ASSERT_SUCCESS(Mc2GenTaskUtils::CreateAicpuTaskV1(context, aicpu_task));
        tasks.insert(tasks.begin() + aicore_idx, aicpu_task);
        return ge::GRAPH_SUCCESS;
    }

    return Mc2GenTaskUtils::Mc2GenTaskCallBack910A2(context, tasks);
}

static ge::Status MatmulAllReduceAddRmsNormCalcOpParam(gert::ExeResGenerationContext* context)
{
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status MatmulAllReduceAddRmsNormGenTask(
    const gert::ExeResGenerationContext* context, std::vector<std::vector<uint8_t>>& tasks)
{
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, MatmulAllReduceGenTaskCallback);
}

IMPL_OP_CT(MatmulAllReduceAddRmsNorm).CalcOpParam(MatmulAllReduceAddRmsNormCalcOpParam).GenerateTask(MatmulAllReduceAddRmsNormGenTask);
} // namespace ops
