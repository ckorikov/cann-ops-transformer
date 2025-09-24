# #
# # Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# # This file is a part of the CANN Open Software.
# # Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# # Please refer to the License for details. You may not use this file except in compliance with the License.
# # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# # See LICENSE in the root of the software repository for the full text of the License.
# #/
# message(STATUS "=== Debug: start ops.rope.rope_with_sin_cos_cache.op_host.CMakeLists.txt ")

# add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE rope_with_sin_cos_cache ACLNNTYPE aclnn_exclude)

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        rope_with_sin_cos_cache_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    add_ops_compile_options(
        OP_NAME RopeWithSinCosCache
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_modules_sources(OPTYPE rope_with_sin_cos_cache ACLNNTYPE aclnn_exclude)
endif()