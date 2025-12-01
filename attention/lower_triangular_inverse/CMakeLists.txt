# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
# op_host_aclnnInner  如果自己实现了aclnn接口使用这个
# op_host_aclnn  使用自动生成的aclnn接口
add_ops_compile_options(
        OP_NAME FusedInferAttentionSink
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(lower_triangular_inverse_depends transformer/attention/lower_triangular_inverse PARENT_SCOPE)
target_sources(op_host_aclnnInner PRIVATE
        op_host/lower_triangular_inverse_def.cpp
)

target_sources(optiling PRIVATE
        op_host/op_tiling/lower_triangular_inverse_tiling.cpp
        op_host/op_tiling/lower_triangular_inverse_base_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
    op_host/op_tiling/lower_triangular_inverse_tiling.cpp
    op_host/op_tiling/lower_triangular_inverse_base_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
)

target_sources(opsproto PRIVATE
        op_host/lower_triangular_inverse_infershape.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
)

target_include_directories(optiling PRIVATE)
target_sources(opapi PRIVATE
        op_api/aclnn_lower_triangular_inverse.cpp
        op_api/lower_triangular_inverse.cpp
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_lower_triangular_inverse.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
