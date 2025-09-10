# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

function(add_optiling_modules OP_TILING_MODULE_NAME)

    # set variables
    set(UT_COMMON_INC ${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/common)

    # add op tiling ut common obj
    add_library(${OP_TILING_MODULE_NAME}_common_obj OBJECT)
    add_dependencies(${OP_TILING_MODULE_NAME}_common_obj json)

    # add op tiling ut test cases obj
    add_library(${OP_TILING_MODULE_NAME}_cases_obj OBJECT)
    add_dependencies(${OP_TILING_MODULE_NAME}_cases_obj json)
    target_include_directories(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
        ${UT_COMMON_INC}
        ${JSON_INCLUDE}
        ${GTEST_INCLUDE}
        ${OPBASE_INC_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/common/inc
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/include/exe_graph
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef/exe_graph
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef/graph
    )
    target_link_libraries(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        gtest
    )

    # add op tiling ut static lib
    add_library(${OP_TILING_MODULE_NAME}_static_lib STATIC)
    target_link_libraries(${OP_TILING_MODULE_NAME}_static_lib PRIVATE
        ${OP_TILING_MODULE_NAME}_common_obj
        ${OP_TILING_MODULE_NAME}_cases_obj
    )
endfunction()

function(add_infershape_modules OP_INFERSHAPE_MODULE_NAME)
    # set variables
    set(UT_COMMON_INC ${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/common)

    # add op tiling ut common obj
    add_library(${OP_INFERSHAPE_MODULE_NAME}_common_obj OBJECT)

    # add op tiling ut test cases obj
    add_library(${OP_INFERSHAPE_MODULE_NAME}_cases_obj OBJECT)
    add_dependencies(${OP_INFERSHAPE_MODULE_NAME}_cases_obj json)
    target_include_directories(${OP_INFERSHAPE_MODULE_NAME}_cases_obj PRIVATE
        ${UT_COMMON_INC}
        ${JSON_INCLUDE}
        ${GTEST_INCLUDE}
        ${OPBASE_INC_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/common/inc
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/include/exe_graph
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef/exe_graph
        ${ASCEND_DIR}/opensdk/opensdk/include/metadef/graph
    )
    target_link_libraries(${OP_INFERSHAPE_MODULE_NAME}_cases_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        metadef
        graph
        gtest
    )

    # add op tiling ut static lib
    add_library(${OP_INFERSHAPE_MODULE_NAME}_static_lib STATIC)
    target_link_libraries(${OP_INFERSHAPE_MODULE_NAME}_static_lib PRIVATE
        ${OP_INFERSHAPE_MODULE_NAME}_common_obj
        ${OP_INFERSHAPE_MODULE_NAME}_cases_obj
    )
endfunction()

function(add_modules_ut_sources)
    set(options OPTION_RESERVED)
    set(oneValueArgs HOSTNAME MODE DIR)
    set(multiValueArgs MULIT_RESERVED)

    cmake_parse_arguments(MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    message(STATUS "=== Debug<add_modules_llt_sources>: ${MODULE_HOSTNAME} ${MODULE_MODE} ${MODULE_DIR}")

    if (TARGET ${MODULE_HOSTNAME}_cases_obj)
        string(FIND "${MODULE_HOSTNAME}_cases_obj" "tiling" TILING_FOUND_INDEX)
        if(${TILING_FOUND_INDEX} GREATER_EQUAL 0)
            file(GLOB OPHOST_TILING_SRCS ${MODULE_DIR}/test_*_tiling.cpp)
            target_sources(${MODULE_HOSTNAME}_cases_obj ${MODULE_MODE} ${OPHOST_TILING_SRCS})
            message(STATUS "=== Debug<add_modules_llt_sources>: ${MODULE_HOSTNAME}_cases_obj ${OPHOST_TILING_SRCS}")
        endif()

        string(FIND "${MODULE_HOSTNAME}_cases_obj" "infershape" INFERSHAPE_FOUND_INDEX)
        if(${INFERSHAPE_FOUND_INDEX} GREATER_EQUAL 0)
            file(GLOB OPHOST_INFERSHAPE_SRCS ${MODULE_DIR}/test_*_infershape.cpp)
            target_sources(${MODULE_HOSTNAME}_cases_obj ${MODULE_MODE} ${OPHOST_INFERSHAPE_SRCS})
            message(STATUS "=== Debug<add_modules_llt_sources>: ${MODULE_HOSTNAME}_cases_obj ${OPHOST_INFERSHAPE_SRCS}")
        endif()
    endif ()
endfunction()