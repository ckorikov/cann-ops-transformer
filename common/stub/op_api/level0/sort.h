/**
<<<<<<< HEAD
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
=======
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
>>>>>>> 95f0a14... index 算子开源迁移
 */

/*!
 * \file sort.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_SORT_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_SORT_OP_H_

#include "opdev/op_executor.h"
#include "opdev/fast_vector.h"

namespace l0op {
const std::tuple<aclTensor*, aclTensor*> Sort(const aclTensor* self, int64_t dim, bool descending, bool stable,
                                              op::DataType indicesType, aclOpExecutor* executor);
}

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_SORT_OP_H_
