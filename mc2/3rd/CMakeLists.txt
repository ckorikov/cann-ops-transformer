# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(BUILD_OPEN_PROJECT)
    file(GLOB MC2_3RD_SRCS 
        ${CMAKE_CURRENT_SOURCE_DIR}/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_base_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/batch_mat_mul_v3/op_host/batch_mat_mul_v3_def.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_advanced.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/op_tiling/matmul_v3_l2_cache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/mat_mul_v3_def.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/op_host/op_tiling/debug_tiling.cpp
    )
    if(MC2_3RD_SRCS)
        add_tiling_modules()
        target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${MC2_3RD_SRCS})
    endif()
endif()

add_subdirectory(common/op_host)