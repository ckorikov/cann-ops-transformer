/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_all_all_gather_batch_mat_mul_gen_task.cpp
 * \brief
 */
#include <vector>
#include <map>
#include <set>
#include <string>

#include "checker.h"
#include "mc2_gen_task_utils.h"
#include "runtime/rt_model.h"
#include "runtime/kernel.h"
#include "graph/utils/args_format_desc_utils.h"
#include "framework/common/taskdown_common.h"
#include "op_mc2.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"
#include "register/op_ct_impl_registry.h"
#include "register/op_ext_gentask_registry.h"

namespace ops {
const std::string ALL_TO_ALL_ALL_GATHER_BMM_OP_TYPE = "AlltoAllAllGatherBatchMatMul";
const int32_t GROUP_CNT_OF_MOE_DISTRIBUTE = 2;
const int32_t ONE_GROUP_CNT_OF_MOE_DISTRIBUTE = 1;
const int32_t GROUP_CNT_OF_DISTRIBUTE_BARRIER = 1;
const int32_t GROUP_CNT = 2;
const int32_t MAX_GROUP_CNT = 16;

// key: op type
// value: group cnt
static const std::map<const std::string, int32_t> GROUP_CNT_MAP{
    {ALL_TO_ALL_ALL_GATHER_BMM_OP_TYPE, GROUP_CNT},
};

static const std::unordered_set<std::string> NO_AI_CPU_SET{};

// 对已有结构的重复定义，只在本文件插入 aicpu desc 的时候使用
struct HcclCommParamDescTmp {
    uint64_t version   : 4;
    uint64_t groupNum  : 4;
    uint64_t hasFfts   : 1;
    uint64_t tilingOff : 7;
    uint64_t isDyn     : 48;
};

static bool IsPlatform910B(const char *nodeName)
{
    fe::PlatFormInfos platform_info;
    fe::OptionalInfos optional_info;
    OPS_ERR_IF(fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
                   ge::GRAPH_SUCCESS,
               OPS_LOG_E(nodeName, "Cannot get platform info!"), return false);
    static std::set<std::string> supported_soc = {"Ascend910B"};
    std::string short_soc_version;
    if (!platform_info.GetPlatformRes("version", "Short_SoC_version", short_soc_version) || short_soc_version.empty()) {
        OPS_LOG_E(nodeName, "Cannot get short soc version!");
        return false;
    }
    OPS_LOG_D(nodeName, "Get soc version: %s", short_soc_version.c_str());
    return supported_soc.count(short_soc_version) > 0;
}

static bool GetGroupCnt(const char *nodeName, const char *opType, int32_t &cnt)
{
    const std::string opTypeStr = opType;
    if (GROUP_CNT_MAP.find(opTypeStr) == GROUP_CNT_MAP.end()) {
        OPS_LOG_E(nodeName, "Op type [%s] has not registe in group cnt map.", opType);
        return false;
    }
    cnt = GROUP_CNT_MAP.at(opTypeStr);
    if ((cnt <= 0) || (cnt > MAX_GROUP_CNT)) {
        OPS_LOG_E(nodeName, "Group cnt [%d] is invalid, it should in [1, %d].", cnt, MAX_GROUP_CNT);
        return false;
    }

    return true;
}

static ge::Status Mc2MoeCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status McMoeInsertHiddenInputForAicore(const gert::ExeResGenerationContext *context, const int32_t groupCnt,
                                           domi::TaskDef &aicoreTask)
{
    const char *nodeName = context->GetNodeName();
    // 获取 aicore task
    auto mixl2TaskDef = aicoreTask.mutable_ffts_plus_task();
    GE_ASSERT_NOTNULL(mixl2TaskDef);

    // 获取 mix ctx idx
    const int32_t ctxNum = mixl2TaskDef->ffts_plus_ctx_size();
    int32_t mixCtxIdx;
    for (mixCtxIdx = 0; mixCtxIdx < ctxNum; ++mixCtxIdx) {
        if (mixl2TaskDef->ffts_plus_ctx(mixCtxIdx).op_type() != domi::FftsPlusCtxDef::ATOMIC) {
            break;
        }
    }
    OPS_ERR_IF(mixCtxIdx == ctxNum, OPS_LOG_E(nodeName, "No valid mixl2 context in [%d] context(s).", ctxNum),
               return ge::GRAPH_FAILED);

    // 获取 context
    auto fftsPlusCtx = mixl2TaskDef->mutable_ffts_plus_ctx(mixCtxIdx);
    GE_ASSERT_NOTNULL(fftsPlusCtx);
    auto mixl2TaskCtx = fftsPlusCtx->mutable_mix_aic_aiv_ctx();
    GE_ASSERT_NOTNULL(mixl2TaskCtx);

    // 获取参数
    std::vector<ge::ArgDesc> argDescs;
    const std::string argsFormat = mixl2TaskCtx->args_format();
    OPS_ERR_IF(ge::ArgsFormatDescUtils::Parse(argsFormat, argDescs) != ge::GRAPH_SUCCESS || argDescs.empty(),
               OPS_LOG_E(nodeName, "Failed to parse, argsFormat:[%s]", argsFormat.c_str()), return ge::GRAPH_FAILED);

    // 找到插入位置
    const auto getIdxFunc = [](const std::vector<ge::ArgDesc> &argDescs) {
        size_t insertIdx = 0U; // 从 0: ffts 开始查找
        for (; insertIdx < argDescs.size(); ++insertIdx) {
            if (argDescs[insertIdx].addr_type == ge::AddrType::INPUT) {
                break;
            }
        }
        return insertIdx;
    };

    OPS_LOG_D(nodeName, "Before insert hidden input, task addr size is %u, args format size is %zu.",
              static_cast<uint32_t>(mixl2TaskCtx->task_addr_size()), argDescs.size());
    if (static_cast<uint32_t>(mixl2TaskCtx->task_addr_size()) != argDescs.size()) {
        OPS_LOG_E(nodeName,
                  "Task addr size is not equal with args format size, task addr size %u, args format size %zu.",
                  static_cast<uint32_t>(mixl2TaskCtx->task_addr_size()), argDescs.size());
        return ge::GRAPH_FAILED;
    }

    OPS_ERR_IF(Mc2GenTaskUtils::InsertHiddenInputsForAicoreTaskDef(context, *mixl2TaskCtx, getIdxFunc, groupCnt) !=
                   ge::GRAPH_SUCCESS,
               OPS_LOG_E(context->GetNodeName(), "Failed to insert hidden input for mix task."),
               return ge::GRAPH_FAILED);

    mixl2TaskDef->set_addr_size(mixl2TaskDef->addr_size() + groupCnt);
    OPS_LOG_D(nodeName, "Modify AICore task for mc2 node successfully.");
    OPS_LOG_D(nodeName, "After insert hidden input, task addr size is %u.",
              static_cast<uint32_t>(mixl2TaskCtx->task_addr_size()));
    return ge::GRAPH_SUCCESS;
}

ge::Status McMoeInsertHiddenInputForAicoreV2(const gert::ExeResGenerationContext *context, const int32_t groupCnt,
                                             domi::TaskDef &aicoreTask)
{
    const char *nodeName = context->GetNodeName();
    // 获取 aicore context
    domi::KernelContext *kernel_context;
    if (aicoreTask.type() == RT_MODEL_TASK_KERNEL) {
        auto kernel_def = aicoreTask.mutable_kernel();
        GE_ASSERT_NOTNULL(kernel_def);
        kernel_context = kernel_def->mutable_context();
    } else if (aicoreTask.type() == RT_MODEL_TASK_ALL_KERNEL) {
        auto kernel_with_handle = aicoreTask.mutable_kernel_with_handle();
        GE_ASSERT_NOTNULL(kernel_with_handle);
        kernel_context = kernel_with_handle->mutable_context();
    } else {
        OPS_LOG_E(nodeName, "Invalid task type [%d].", aicoreTask.type());
        return ge::GRAPH_FAILED;
    }
    GE_ASSERT_NOTNULL(kernel_context);

    // 获取参数
    std::vector<ge::ArgDesc> argDescs;
    const std::string argsFormat = kernel_context->args_format();
    OPS_ERR_IF(ge::ArgsFormatDescUtils::Parse(argsFormat, argDescs) != ge::GRAPH_SUCCESS || argDescs.empty(),
               OPS_LOG_E(nodeName, "Failed to parse, argsFormat:[%s]", argsFormat.c_str()), return ge::GRAPH_FAILED);

    // 找到插入位置
    const auto getIdxFunc = [](const std::vector<ge::ArgDesc> &argDescs) {
        size_t insertIdx = 0U; // 从 0: ffts 开始查找
        for (; insertIdx < argDescs.size(); ++insertIdx) {
            if ((argDescs[insertIdx].addr_type == ge::AddrType::INPUT) ||
                (argDescs[insertIdx].addr_type == ge::AddrType::INPUT_INSTANCE)) {
                break;
            }
        }
        return insertIdx;
    };

    OPS_ERR_IF(Mc2GenTaskUtils::InsertHiddenInputsForAicoreTaskDefV2(context, *kernel_context, getIdxFunc, groupCnt) !=
                   ge::GRAPH_SUCCESS,
               OPS_LOG_E(context->GetNodeName(), "Failed to insert hidden input for mix task."),
               return ge::GRAPH_FAILED);

    OPS_LOG_D(nodeName, "Modify AICore task for mc2 node successfully.");
    return ge::GRAPH_SUCCESS;
}

ge::Status CreateAicpuTaskMc2Moe(const gert::ExeResGenerationContext *context, domi::TaskDef &aicpuTask,
                                 const int32_t groupCnt)
{
    // desc 配置
    union {
        HcclCommParamDescTmp hcclCommParamDesc;
        uint64_t customValue;
    } desc;
    desc.hcclCommParamDesc.version = 1; // 保留参数，暂时保持为 1
    desc.hcclCommParamDesc.groupNum = groupCnt;
    desc.hcclCommParamDesc.hasFfts = 0; // aicpu 暂时用不到此参数
    desc.hcclCommParamDesc.tilingOff = desc.hcclCommParamDesc.hasFfts + desc.hcclCommParamDesc.groupNum + 1;
    desc.hcclCommParamDesc.isDyn = 0; // 现在默认为 0

    // aicpu 参数顺序: {desc}{ffts}{hcom}{tiling}
    std::vector<ge::ArgDesc> argsDes;
    ge::ArgsFormatDescUtils::InsertCustomValue(argsDes, 0, desc.customValue);
    if (desc.hcclCommParamDesc.hasFfts) {
        ge::ArgsFormatDescUtils::Append(argsDes, ge::AddrType::FFTS_ADDR);
    }
    ge::ArgsFormatDescUtils::InsertHiddenInputs(argsDes, -1, ge::HiddenInputsType::HCOM, groupCnt);
    ge::ArgsFormatDescUtils::Append(argsDes, ge::AddrType::TILING);

    // 设置 so、kernel name
    const std::string soName = "libccl_kernel.so";
    const std::string kernelName = "RunAicpuKfcSrvLaunch";
    const std::string argsFormat = ge::ArgsFormatDescUtils::ToString(argsDes);

    return Mc2GenTaskUtils::CreateAicpuTask(context, soName, kernelName, argsFormat, aicpuTask);
}

// 单、双通信域均需要保证填入task顺序 [wait task, aicpu task, record task, aicore task]
ge::Status Mc2MoeInsertTask(const gert::ExeResGenerationContext *context, std::vector<domi::TaskDef> &tasks,
                            int64_t &aicoreIndex, int32_t groupCnt, const char *opType)
{
    domi::TaskDef waitTask{};
    const char *nodeName = context->GetNodeName();
    OPS_ERR_IF(Mc2GenTaskUtils::CreateNotifyTask(context, waitTask, RT_MODEL_TASK_NOTIFY_WAIT, true) !=
                   ge::GRAPH_SUCCESS,
               OPS_LOG_E(context->GetNodeName(), "Failed to create notify wait task."), return false);
    const std::string opTypeStr = opType;
    if (opTypeStr == DISTRIBUTE_BARRIER_OP_TYPE) {
        waitTask.set_private_def("group", sizeof("group"));
    } else {
        waitTask.set_private_def("group_ep", sizeof("group_ep"));
    }

    tasks.insert(tasks.begin() + aicoreIndex, waitTask);
    OPS_LOG_EVENT(nodeName, "Generate notify wait task for mc2 node successfully.");
    aicoreIndex += 1L;
    bool needAicpuTesk = IsPlatform910B(nodeName) || NO_AI_CPU_SET.find(opTypeStr) == NO_AI_CPU_SET.end();
    if (needAicpuTesk) {
        domi::TaskDef aicpuTask{};
        if (CreateAicpuTaskMc2Moe(context, aicpuTask, groupCnt) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (opType == MOE_DISTRIBUTE_DISPATCH_SETUP_OP_TYPE || opType == MOE_DISTRIBUTE_COMBINE_SETUP_OP_TYPE) {
            aicpuTask.mutable_kernel()->set_block_dim(4U);
            OPS_LOG_D(nodeName, "Set aicpu blockdim");
        }
        tasks.insert(tasks.begin() + aicoreIndex, aicpuTask);
        OPS_LOG_D(nodeName, "Generate aicpu task for mc2 node successfully, proto = %s.",
                  aicpuTask.DebugString().c_str());
        aicoreIndex += 1L;
    }
    domi::TaskDef recordTask{};
    OPS_ERR_IF(Mc2GenTaskUtils::CreateNotifyTask(context, recordTask, RT_MODEL_TASK_NOTIFY_RECORD, false) !=
                   ge::GRAPH_SUCCESS,
               OPS_LOG_E(context->GetNodeName(), "Failed to create notify record task."), return false);

    if (opTypeStr == DISTRIBUTE_BARRIER_OP_TYPE) {
        recordTask.set_private_def("group", sizeof("group"));
    } else {
        recordTask.set_private_def("group_ep", sizeof("group_ep"));
    }
    tasks.insert(tasks.begin() + aicoreIndex, recordTask);
    OPS_LOG_D(nodeName, "Generate notify record task for mc2 node successfully.");
    aicoreIndex += 2L; // 跳过aicore task

    OPS_LOG_EVENT(nodeName, "Generate task for mc2 node successfully.");
    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2MoeGenTaskCallback(const gert::ExeResGenerationContext *context, std::vector<domi::TaskDef> &tasks)
{
    const char *nodeName = context->GetNodeName();
    int64_t aicoreIndex = Mc2GenTaskUtils::GetTaskIdxByType(context, tasks, RT_MODEL_TASK_FFTS_PLUS_TASK);
    OPS_ERR_IF(aicoreIndex < 0, OPS_LOG_E(nodeName, "Failed to get AICore task."), return ge::GRAPH_FAILED);
    OPS_LOG_D(nodeName, "Start to generate task for MC2, task def size [%lu], aicore index [%ld].", tasks.size(),
              aicoreIndex);

    int32_t groupCnt = -1;
    const char *opType = context->GetNodeType();
    if (opType == nullptr) {
        OPS_LOG_E(nodeName, "Op type is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (!GetGroupCnt(nodeName, opType, groupCnt)) {
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(nodeName, "Op [%s] get group [%d] success.", opType, groupCnt);

    if (McMoeInsertHiddenInputForAicore(context, groupCnt, tasks[static_cast<size_t>(aicoreIndex)]) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Insert hidden input for [%s] failed.", opType);
        return ge::GRAPH_FAILED;
    }

    if (Mc2GenTaskUtils::IsComputationOnly()) {
        OPS_LOG_D(nodeName, "Debug mode is 1 (i.e. computation only), generating task for mc2 node ends.");
        return ge::GRAPH_SUCCESS;
    }

    const std::string opTypeStr = opType;
    if (opTypeStr == MOE_DISTRIBUTE_COMBINE_TEARDOWN_OP_TYPE || opTypeStr == MOE_DISTRIBUTE_DISPATCH_TEARDOWN_OP_TYPE) {
        return ge::GRAPH_SUCCESS;
    } else {
        return Mc2MoeInsertTask(context, tasks, aicoreIndex, groupCnt, opType);
    }
}

// 支持静态图在线编译.o
ge::Status Mc2MoeGenTaskCallbackV2(const gert::ExeResGenerationContext *context, std::vector<domi::TaskDef> &tasks)
{
    const char *nodeName = context->GetNodeName();
    int64_t aicoreIndex = -1L;
    for (int64_t i = static_cast<int64_t>(tasks.size()) - 1; i >= 0; i--) { // 倒叙遍历
        if ((tasks[i].type() == RT_MODEL_TASK_KERNEL) || (tasks[i].type() == RT_MODEL_TASK_ALL_KERNEL)) {
            aicoreIndex = i;
            break;
        }
    }
    OPS_ERR_IF(aicoreIndex < 0, OPS_LOG_E(nodeName, "Failed to get AICore task."), return ge::GRAPH_FAILED);
    OPS_LOG_D(nodeName, "Start to generate task for MC2, task def size [%lu], aicore index [%ld].", tasks.size(),
              aicoreIndex);

    int32_t groupCnt = -1;
    const char *opType = context->GetNodeType();
    OPS_ERR_IF(opType == nullptr, OPS_LOG_E(nodeName, "Op type is nullptr."), return ge::GRAPH_FAILED);
    if (!GetGroupCnt(nodeName, opType, groupCnt)) {
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_EVENT(nodeName, "Op [%s] get group [%d] success.", opType, groupCnt);

    if (McMoeInsertHiddenInputForAicoreV2(context, groupCnt, tasks[static_cast<size_t>(aicoreIndex)]) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Insert hidden input for [%s] failed.", opType);
        return ge::GRAPH_FAILED;
    }

    if (Mc2GenTaskUtils::IsComputationOnly()) {
        OPS_LOG_D(nodeName, "Debug mode is 1 (i.e. computation only), generating task for mc2 node ends.");
        return ge::GRAPH_SUCCESS;
    }

    const std::string opTypeStr = opType;
    bool useAiCpu = ((opTypeStr != MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE) &&
                     (opTypeStr != MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE) && (opTypeStr != DISTRIBUTE_BARRIER_OP_TYPE) &&
                     (opTypeStr != MOE_DISTRIBUTE_DISPATCH_OP_TYPE) && (opTypeStr != MOE_DISTRIBUTE_COMBINE_OP_TYPE));
    return useAiCpu ? Mc2MoeInsertTask(context, tasks, aicoreIndex, groupCnt, opType) : ge::GRAPH_SUCCESS;
}

ge::Status Mc2MoeGenTaskFunc(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks)
{
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2MoeGenTaskCallback);
}

ge::Status Mc2MoeGenTaskFuncV2(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_EVENT(nodeName, "MC2 Generate task start.");
    if (IsPlatform910B(nodeName)) {
        return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2MoeGenTaskCallback);
    }
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2MoeGenTaskCallbackV2);
}

IMPL_OP_CT(AlltoAllAllGatherBatchMatMul).CalcOpParam(Mc2MoeCalcParamFunc).GenerateTask(Mc2MoeGenTaskFunc);
} // namespace ops
