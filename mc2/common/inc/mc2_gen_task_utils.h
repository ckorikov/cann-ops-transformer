/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFERINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_gen_task_utils.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_RUNTIME_MC2_GEN_TASK_UTILS
#define OPS_BUILT_IN_OP_PROTO_RUNTIME_MC2_GEN_TASK_UTILS

#include "runtime/rt_model.h"
#include "proto/task.pb.h"
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/utils/args_format_desc_utils.h"

namespace ops {
using GenTaskFunc = ge::Status (*)(const gert::ExeResGenerationContext *, std::vector<domi::TaskDef> &);
class Mc2GenTaskUtils {
public:
    static bool IsComputationOnly();
    static ge::Status CommonKFCMc2CalcParamFunc(
        gert::ExeResGenerationContext *context, const ge::AscendString &name, const ge::AscendString &reuse_key);
    static ge::Status CommonKFCMc2GenTask(
        const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks, GenTaskFunc func);
    static int64_t GetTaskIdxByType(
        const gert::ExeResGenerationContext *context, const std::vector<domi::TaskDef> &tasks, rtModelTaskType_t type);
    static ge::Status InsertHiddenInputsForAicoreTaskDef(const gert::ExeResGenerationContext *context,
        domi::FftsPlusMixAicAivCtxDef &ctx_def, size_t (*get_insert_idx)(const std::vector<ge::ArgDesc> &),
        size_t input_cnt = 1U);
    static ge::Status InsertHiddenInputsForAicoreTaskDefV2(const gert::ExeResGenerationContext *context,
        domi::KernelContext &ctx_def, size_t (*get_insert_idx)(const std::vector<ge::ArgDesc> &), size_t input_cnt);
    static ge::Status InsertHiddenInputForAicoreV1(
        const gert::ExeResGenerationContext *context, domi::TaskDef &aicore_task);
    static ge::Status CreateAicpuTask(const gert::ExeResGenerationContext *context, const std::string &so_name,
        const std::string &kernel_name, const std::string &args_format, domi::TaskDef &aicpu_task);
    static ge::Status CreateAicpuTaskV1(const gert::ExeResGenerationContext *context, domi::TaskDef &aicpu_task);
    static ge::Status CreateNotifyTask(const gert::ExeResGenerationContext *context, domi::TaskDef &notify_task,
        rtModelTaskType_t type, bool is_attached_stream);
    static ge::Status Mc2GenTaskCallBack910A2(
        const gert::ExeResGenerationContext *context, std::vector<domi::TaskDef> &tasks);
};
}  // namespace ops
#endif  // OPS_BUILT_IN_OP_PROTO_RUNTIME_MC2_GEN_TASK_UTILS
