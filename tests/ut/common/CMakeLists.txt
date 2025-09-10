# Copyright(c) Huawei Technologies Co., Ltd.2025. All rights reserved.
# This File is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License");
# Please refer to the Licence for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

cmake_minimum_required(VERSION 3.16)

if(UT_TEST_ALL OR OP_HOST_UT)
    file(GLOB OP_TILING_UT_COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_run_context_faker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_cube_util.cpp
    )
    target_sources(${OP_TILING_MODULE_NAME}_common_obj PRIVATE
        ${OP_TILING_UT_COMMON_SRC}
    )
    target_include_directories(${OP_TILING_MODULE_NAME}_common_obj PRIVATE
        ${JSON_INCLUDE}
        ${ASCEND_DIR}/include/exe_graph
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef/graph
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef/exe_graph
    )
    target_link_libraries(${OP_TILING_MODULE_NAME}_common_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        c_sec
    )

    file(GLOB OP_INFERSHAPE_UT_COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/infershape_test_util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ut_op_util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ut_op_common.cpp
    )
    target_sources(${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE
        ${OP_INFERSHAPE_UT_COMMON_SRC}
    )
    target_include_directories(${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE
        ${ASCEND_DIR}/include/exe_graph
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef/graph
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef/exe_graph
    )
    target_link_libraries(${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        c_sec
    )
endif()
