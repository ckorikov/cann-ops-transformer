/* *
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

/* !
 * \file all_gather_matmul_gen_task.cc
 * \brief
 */
#include "register/op_ct_impl_registry.h"
#include "mc2_gen_task_utils.h"

namespace ops {
static ge::Status AllGatherMatmulGenTaskCallback(const gert::ExeResGenerationContext *context,
                                                 std::vector<domi::TaskDef> &tasks) {
    return Mc2GenTaskUtils::Mc2GenTaskCallBack910A2(context, tasks);
}

static ge::Status AllGatherMatmulCalcOpParam(gert::ExeResGenerationContext *context) {
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

static ge::Status AllGatherMatmulGenTask(const gert::ExeResGenerationContext *context,
                                         std::vector<std::vector<uint8_t>> &tasks) {
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, AllGatherMatmulGenTaskCallback);
}

IMPL_OP_CT(AllGatherMatmul).CalcOpParam(AllGatherMatmulCalcOpParam).GenerateTask(AllGatherMatmulGenTask);
} // namespace ops