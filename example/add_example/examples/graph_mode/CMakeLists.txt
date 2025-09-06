# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

# 设置工程名
project(GE_IR_EXAMPLE)

if(NOT "$ENV{ASCEND_OPP_PATH}" STREQUAL "")
    get_filename_component(ASCEND_PATH $ENV{ASCEND_OPP_PATH} DIRECTORY)
elseif(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/latest")
endif()

set(FWK_INCLUDE_DIR "${ASCEND_PATH}/compiler/include")

message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

file(GLOB files CONFIGURE_DEPENDS
    test_geir_add_example.cpp
)

add_executable(test_geir_add_example ${files})

find_library(GRAPH_LIBRARY_DIR libgraph.so "${ASCEND_PATH}/compiler/lib64/stub")
find_library(GE_RUNNER_LIBRARY_DIR libge_runner.so "${ASCEND_PATH}/compiler/lib64/stub")
find_library(GRAPH_BASE_LIBRARY_DIR libgraph_base.so "${ASCEND_PATH}/compiler/lib64")

target_link_libraries(test_geir_add_example PRIVATE
    ${GRAPH_LIBRARY_DIR}
    ${GE_RUNNER_LIBRARY_DIR}
    ${GRAPH_BASE_LIBRARY_DIR}
)

target_include_directories(test_geir_add_example PRIVATE
    ${FWK_INCLUDE_DIR}/graph/
    ${FWK_INCLUDE_DIR}/ge/
    ${ASCEND_PATH}/opp/built-in/op_proto/inc/
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_PATH}/compiler/include
)