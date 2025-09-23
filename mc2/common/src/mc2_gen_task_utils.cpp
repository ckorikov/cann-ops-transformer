/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file mc2_gen_task_utils.cc
 * \brief
 */

#include "mc2_gen_task_utils.h"
#include "ascend_string.h"
#include "runtime/rt_model.h"
#include "checker.h"
#include "error/ops_error.h"
#include "error_util.h"
#include "proto/task.pb.h"
#include "framework/common/taskdown_common.h"
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/utils/args_format_desc_utils.h"
#include "register/hidden_inputs_func_registry.h"

namespace {
constexpr int64_t INVALID_INT_VAL = -1;
const std::string SO_NAME = "libccl_kernel.so";
const std::string KERNEL_NAME_V1 = "RunAicpuKfcSrvLaunch";

// 对已有结构的重复定义，只在本文件插入 aicpu desc 的时候使用
struct HcclCommParamDescTemp {
    uint64_t version : 4;
    uint64_t groupNum : 4;
    uint64_t hasFfts : 1;
    uint64_t tilingOff : 7;
    uint64_t isDyn : 48;
};

ge::Status DeserializeTask(const char *node, const std::vector<std::vector<uint8_t>> &tasks,
                           std::vector<domi::TaskDef>& pb_tasks)
{
    OPS_LOG_D(node, "Task size is %zu", tasks.size());
    pb_tasks.clear();
    for (auto &task : tasks) {
        domi::TaskDef pb_task;
        const bool ret = pb_task.ParseFromArray(task.data(), task.size());
        GE_ASSERT_TRUE(ret);
        OPS_LOG_D(node, "Task pb size is %zu, parse ret %d.", task.size(), ret);
        pb_tasks.emplace_back(std::move(pb_task));
    }
    return ge::SUCCESS;
}

ge::Status SerializeTask(const char *node, std::vector<std::vector<uint8_t>> &tasks, std::vector<domi::TaskDef> &pb_tasks)
{
    OPS_LOG_D(node, "After task size is %zu", tasks.size());
    tasks.clear();
    for (auto &task : pb_tasks) {
        const size_t pb_size = task.ByteSizeLong();
        std::vector<uint8_t> pb_buff(pb_size, 0);
        const bool ret = task.SerializeToArray(pb_buff.data(), pb_size);
        GE_ASSERT_TRUE(ret);
        OPS_LOG_D(node, "Task pb size is %zu, parse ret %d.", pb_size, ret);
        tasks.emplace_back(std::move(pb_buff));
    }
    return ge::SUCCESS;
}

int64_t GetAttachStreamIdByContext(const gert::ExeResGenerationContext *context, size_t idx = 0)
{
#ifndef ASCEND_OPSPROTO_UT
    const auto stream_infos = context->GetAttachedStreamInfos();
    if (idx >= stream_infos.size()) {
        OPS_LOG_E(context->GetNodeName(), "Invalid index %zu in streams count %zu.", idx, stream_infos.size());
        return INVALID_INT_VAL;
    }

    const int64_t stream_id = (stream_infos[0].is_valid ? stream_infos[0].stream_id : INVALID_INT_VAL);
#else
    const int64_t stream_id = 1;
#endif
    return stream_id;
}
}

namespace ops {
bool Mc2GenTaskUtils::IsComputationOnly()
{
    const char *env = getenv("ASCEND_MC2_DEBUG_MODE");
    return (env != nullptr && std::atoi(env) == 1);
}

ge::Status Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(
    gert::ExeResGenerationContext *context, const ge::AscendString &name, const ge::AscendString &reuse_key)
{
    GE_ASSERT_NOTNULL(context);
    gert::StreamInfo stream_info;
    std::vector<int64_t> stream_depend_value(0);
    stream_info.name = name;
    stream_info.reuse_key = reuse_key;
    stream_info.depend_value_input_indices = stream_depend_value;
    stream_info.required = true;

    std::vector<gert::StreamInfo> stream_infos;
    stream_infos.push_back(stream_info);
    const auto ret = context->SetAttachedStreamInfos(stream_infos);
    if (ret != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to set attached stream infos.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

int64_t Mc2GenTaskUtils::GetTaskIdxByType(
    const gert::ExeResGenerationContext *context, const std::vector<domi::TaskDef> &tasks, rtModelTaskType_t type)
{
    GE_ASSERT_NOTNULL(context);
    const size_t task_size = tasks.size();
    OPS_ERR_IF(task_size == 0U, OPS_LOG_E(context->GetNodeName(), "Empty task vector when generating task."),
               return INVALID_INT_VAL);
    for (int64_t i = task_size - 1U; i >= 0; --i) {
        const domi::TaskDef &cur_task = tasks[i];
        if (static_cast<rtModelTaskType_t>(cur_task.type()) == type) {
            return static_cast<int64_t>(i);
        }
    }
    OPS_LOG_W(context->GetNodeName(), "The specific taskdef is not found, type %u.", static_cast<uint32_t>(type));
    return INVALID_INT_VAL;
}

ge::Status Mc2GenTaskUtils::InsertHiddenInputsForAicoreTaskDef(
    const gert::ExeResGenerationContext *context, domi::FftsPlusMixAicAivCtxDef &ctx_def,
    size_t (*get_insert_idx)(const std::vector<ge::ArgDesc> &), size_t input_cnt)
{
    GE_ASSERT_NOTNULL(context);
    GE_ASSERT_NOTNULL(get_insert_idx);

    std::vector<ge::ArgDesc> arg_descs;
    const std::string args_format = ctx_def.args_format();
    OPS_CHECK(ge::ArgsFormatDescUtils::Parse(args_format, arg_descs) != ge::GRAPH_SUCCESS || arg_descs.empty(),
              OPS_LOG_E(context->GetNodeName(), "Failed to parse args_format:[%s]", args_format.c_str()),
              return ge::GRAPH_FAILED);

    const size_t insert_idx = get_insert_idx(arg_descs);
    OPS_LOG_I(context->GetNodeName(), "Origin format is %s, insertion position is %zu.", args_format.c_str(), insert_idx);
    ge::ArgsFormatDescUtils::InsertHiddenInputs(arg_descs, insert_idx, ge::HiddenInputsType::HCOM, input_cnt);
    const std::string new_args_format = ge::ArgsFormatDescUtils::Serialize(arg_descs);
    ctx_def.set_args_format(new_args_format);
    OPS_LOG_I(context->GetNodeName(), "New args format is %s, size is %zu.", new_args_format.c_str(), arg_descs.size());

    const int64_t task_addr_size = ctx_def.task_addr_size();
    OPS_CHECK(task_addr_size < static_cast<int64_t>(insert_idx),
              OPS_LOG_E(context->GetNodeName(), "Invalid task addr size %ld", task_addr_size),
              return ge::GRAPH_FAILED);

    // 1、先从结尾开始添加，添加的数量就是input_cnt，前面一部分添加空地址占位符，后面一部分需要把前面的地址挪过来
    // case1：原来的task addr是1234，插入位置为1，插入数量为1，那么这一步处理之后的结果是1234 4
    // case2：原来的task addr是1234，插入位置为1，插入数量为10，那么这一步处理之后的结果是1234 0000000234
    // case1中的mark_pos为负数，那就说明后面不需要添空地址
    int64_t mark_pos = static_cast<int64_t>(insert_idx + input_cnt) - task_addr_size;
    for (int64_t i = 0; i < static_cast<int64_t>(input_cnt); ++i) {
        ctx_def.add_task_addr(i < mark_pos ? 0UL : ctx_def.task_addr(i + task_addr_size - input_cnt));
    }

    // 2、把从insert_idx开始到原先数组结尾部分处理一下，很显然[insert_idx, insert_idx + input_cnt)这一区间的值是空地址
    // 如果还有剩下的部分，则是取往前input_cnt的元素，需要从后往前遍历
    // case1：1 [234] 4          -> 1 [023] 4
    // case2：1 [234] 0000000234 -> 1 [000] 0000000234
    mark_pos = static_cast<int64_t>(insert_idx + input_cnt);
    for (int64_t i = task_addr_size - 1; i >= static_cast<int64_t>(insert_idx); --i) {
        ctx_def.set_task_addr(i, i < mark_pos ? 0UL : ctx_def.task_addr(i - input_cnt));
    }

    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2GenTaskUtils::InsertHiddenInputsForAicoreTaskDefV2(
    const gert::ExeResGenerationContext *context, domi::KernelContext &ctx_def,
    size_t (*get_insert_idx)(const std::vector<ge::ArgDesc> &), size_t input_cnt)
{
    GE_ASSERT_NOTNULL(context);
    GE_ASSERT_NOTNULL(get_insert_idx);

    std::vector<ge::ArgDesc> arg_descs;
    const std::string args_format = ctx_def.args_format();
    OPS_CHECK(ge::ArgsFormatDescUtils::Parse(args_format, arg_descs) != ge::GRAPH_SUCCESS || arg_descs.empty(),
              OPS_LOG_E(context->GetNodeName(), "Failed to parse args_format:[%s]", args_format.c_str()),
              return ge::GRAPH_FAILED);

    const size_t insert_idx = get_insert_idx(arg_descs);
    OPS_LOG_I(context->GetNodeName(), "Origin format is %s, insertion position is %zu.", args_format.c_str(), insert_idx);
    ge::ArgsFormatDescUtils::InsertHiddenInputs(arg_descs, insert_idx, ge::HiddenInputsType::HCOM, input_cnt);
    const std::string new_args_format = ge::ArgsFormatDescUtils::Serialize(arg_descs);
    ctx_def.set_args_format(new_args_format);
    OPS_LOG_I(context->GetNodeName(), "New args format is %s, size is %zu.", new_args_format.c_str(), arg_descs.size());

    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2GenTaskUtils::InsertHiddenInputForAicoreV1(const gert::ExeResGenerationContext *context,
                                                         domi::TaskDef &aicore_task)
{
    auto mixl2_task_def = aicore_task.mutable_ffts_plus_task();
    GE_ASSERT_NOTNULL(mixl2_task_def);
    const int32_t ctx_num = mixl2_task_def->ffts_plus_ctx_size();
    int32_t mix_ctx_idx = 0;
    for ( ; mix_ctx_idx < ctx_num; ++mix_ctx_idx) {
        if (mixl2_task_def->ffts_plus_ctx(mix_ctx_idx).op_type() != domi::FftsPlusCtxDef::ATOMIC) {
            break;
        }
    }
    OP_CHECK(mix_ctx_idx == ctx_num,
             OPS_LOG_E(context->GetNodeName(), "No valid mixl2 context in %d context(s).", ctx_num),
             return ge::GRAPH_FAILED);

    auto ffts_plus_ctx = mixl2_task_def->mutable_ffts_plus_ctx(mix_ctx_idx);
    GE_ASSERT_NOTNULL(ffts_plus_ctx);
    auto mixl2_task_ctx = ffts_plus_ctx->mutable_mix_aic_aiv_ctx();
    GE_ASSERT_NOTNULL(mixl2_task_ctx);
    const auto get_idx = [](const std::vector<ge::ArgDesc> &args_descs) {
        size_t insert_idx = 0U;
        for (; insert_idx < args_descs.size(); ++insert_idx) {
            if (args_descs[insert_idx].addr_type == ge::AddrType::INPUT) {
                break;
            }
        }
        return insert_idx;
    };
    OP_CHECK(InsertHiddenInputsForAicoreTaskDef(context, *mixl2_task_ctx, get_idx) != ge::GRAPH_SUCCESS,
             OPS_LOG_E(context->GetNodeName(), "Failed to insert hidden input for mix task."),
             return ge::GRAPH_FAILED);
    mixl2_task_def->set_addr_size(mixl2_task_def->addr_size() + 1);
    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2GenTaskUtils::CreateAicpuTask(
    const gert::ExeResGenerationContext *context, const std::string &so_name, const std::string &kernel_name,
    const std::string &args_format, domi::TaskDef &aicpu_task)
{
    GE_ASSERT_NOTNULL(context);
    const int64_t stream_id = GetAttachStreamIdByContext(context);
    if (stream_id < 0) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get valid attached stream id.");
        return ge::GRAPH_FAILED;
    }
    aicpu_task.set_type(RT_MODEL_TASK_KERNEL);
    aicpu_task.set_stream_id(stream_id);
    OPS_LOG_I(context->GetNodeName(), "Set stream id %ld for AICpu task.", stream_id);

    auto kernel_def = aicpu_task.mutable_kernel();
    kernel_def->set_so_name(so_name);
    kernel_def->set_kernel_name(kernel_name);

    auto mutable_context = kernel_def->mutable_context();
    mutable_context->set_kernel_type(static_cast<uint32_t>(ge::ccKernelType::AI_CPU_KFC));
    mutable_context->set_op_index(context->GetOpId());
    mutable_context->set_args_format(args_format);
    OPS_LOG_I(context->GetNodeName(), "Create AICpu task for mc2 node successfully, format %s.", args_format.c_str());
    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2GenTaskUtils::CreateAicpuTaskV1(const gert::ExeResGenerationContext *context, domi::TaskDef &aicpu_task)
{
    size_t inputSize = context->GetComputeNodeInfo()->GetIrInputsNum();
    size_t outputSize = context->GetComputeNodeInfo()->GetIrOutputsNum();
    OPS_LOG_I(context->GetNodeName(), "IR inputNum: %zu, IR outputNum: %zu", inputSize, outputSize);
    const size_t index = 3;
    // desc 配置
    union {
        HcclCommParamDescTemp hcclCommParamDesc;
        uint64_t customValue;
    } desc;
    desc.hcclCommParamDesc.version = 1; // 保留参数，暂时保持为 1
    desc.hcclCommParamDesc.groupNum = 1; // group 只支持1
    desc.hcclCommParamDesc.hasFfts = 0; // aicpu 暂时用不到此参数
    desc.hcclCommParamDesc.tilingOff = index + inputSize + outputSize; // tiling 在args中的index
    desc.hcclCommParamDesc.isDyn = 0; // 现在默认为 0

    // aicpu 参数顺序: {desc}{ffts}{hcom}{INPUT0}...{INPUTN}{OUTPUT0}...{OUTPUTN}{WORKSPACE}{TILING}
    // ffts 目前没有
    std::vector<ge::ArgDesc> args;
    ge::ArgsFormatDescUtils::InsertCustomValue(args, 0, desc.customValue);
    ge::ArgsFormatDescUtils::InsertHiddenInputs(args, -1, ge::HiddenInputsType::HCOM);
    ge::ArgsFormatDescUtils::Append(args, ge::AddrType::INPUT, 0);
    for (size_t i = 1; i < inputSize; i++) {
        ge::ArgsFormatDescUtils::Append(args, ge::AddrType::PLACEHOLDER);
    }
    for (size_t j = 0; j < outputSize; j++) {
        ge::ArgsFormatDescUtils::Append(args, ge::AddrType::OUTPUT, j);
    }
    ge::ArgsFormatDescUtils::Append(args, ge::AddrType::WORKSPACE, 0);
    ge::ArgsFormatDescUtils::Append(args, ge::AddrType::TILING);
    return CreateAicpuTask(context, SO_NAME, KERNEL_NAME_V1, ge::ArgsFormatDescUtils::ToString(args), aicpu_task);
}

ge::Status Mc2GenTaskUtils::CommonKFCMc2GenTask(const gert::ExeResGenerationContext *context,
                                                std::vector<std::vector<uint8_t>> &tasks, GenTaskFunc func)
{
    GE_ASSERT_NOTNULL(context);
    GE_ASSERT_NOTNULL(func);
    std::vector<domi::TaskDef> pb_tasks;
    const char *node_name = context->GetNodeName();
    GE_ASSERT_SUCCESS(DeserializeTask(node_name, tasks, pb_tasks));
    const ge::Status ret = func(context, pb_tasks);
    GE_ASSERT_SUCCESS(SerializeTask(node_name, tasks, pb_tasks));
    return ret;
}

ge::Status Mc2GenTaskUtils::CreateNotifyTask(const gert::ExeResGenerationContext *context, domi::TaskDef &notify_task,
                                             rtModelTaskType_t type, bool is_attached_stream)
{
    GE_ASSERT_NOTNULL(context);
    int64_t stream_id;
    if (is_attached_stream) {
        stream_id = GetAttachStreamIdByContext(context);
    } else {
        stream_id = context->GetStreamId();
    }
    GE_ASSERT_TRUE(stream_id >= 0);
    notify_task.set_id(context->GetOpId());
    notify_task.set_notify_id(UINT32_MAX);
    notify_task.set_type(type);
    notify_task.set_stream_id(stream_id);
    OPS_LOG_I(context->GetNodeName(), "Create notify task(type %u) for mc2 node successfully, %s stream id %ld.",
              static_cast<uint32_t>(type), (is_attached_stream ? "attached" : "main"), stream_id);
    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2GenTaskUtils::Mc2GenTaskCallBack910A2(const gert::ExeResGenerationContext *context,
                                                    std::vector<domi::TaskDef> &tasks)
{
    int64_t aicore_idx = GetTaskIdxByType(context, tasks, RT_MODEL_TASK_FFTS_PLUS_TASK);
    OP_CHECK(aicore_idx < 0, OPS_LOG_E(context->GetNodeName(), "Failed to get AICore task."), return ge::GRAPH_FAILED);
    OPS_LOG_I(context->GetNodeName(), "Start to generate task for MC2, task def size %lu, aicore index %ld.",
              tasks.size(), aicore_idx);

    /* wait aicpu record [aicore] wait */
    domi::TaskDef aicpu_wait_for_aicore{};
    GE_ASSERT_SUCCESS(CreateNotifyTask(context, aicpu_wait_for_aicore, RT_MODEL_TASK_NOTIFY_WAIT, true));
    tasks.insert(tasks.begin() + aicore_idx, aicpu_wait_for_aicore);
    ++aicore_idx;

    domi::TaskDef aicpu_task{};
    GE_ASSERT_SUCCESS(CreateAicpuTaskV1(context, aicpu_task));
    tasks.insert(tasks.begin() + aicore_idx, aicpu_task);
    ++aicore_idx;

    domi::TaskDef aicore_record_for_aicpu{};
    GE_ASSERT_SUCCESS(CreateNotifyTask(context, aicore_record_for_aicpu, RT_MODEL_TASK_NOTIFY_RECORD, false));
    tasks.insert(tasks.begin() + aicore_idx, aicore_record_for_aicpu);
    ++aicore_idx;

    return InsertHiddenInputForAicoreV1(context, tasks[static_cast<size_t>(aicore_idx)]);
}
} // namespace ops
