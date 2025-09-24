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

/* !
 * \file matmul_reduce_scatter_gen_task.cpp
 * \brief
 */
#include "register/op_ct_impl_registry.h"
#include "mc2_gen_task_utils.h"

namespace ops {
static ge::Status MatmulReduceScatterGenTaskCallback(const gert::ExeResGenerationContext *context,
                                                     std::vector<domi::TaskDef> &tasks) {
  return Mc2GenTaskUtils::Mc2GenTaskCallBack910A2(context, tasks);
}

static ge::Status MatmulReduceScatterCalcOpParam(gert::ExeResGenerationContext *context) {
  return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

static ge::Status MatmulReduceScatterGenTask(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks) {
  return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, MatmulReduceScatterGenTaskCallback);
}

IMPL_OP_CT(MatmulReduceScatter).CalcOpParam(MatmulReduceScatterCalcOpParam).GenerateTask(MatmulReduceScatterGenTask);
} // namespace ops