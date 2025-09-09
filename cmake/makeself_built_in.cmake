# Copyright(c) Huawei Technologies Co., Ltd.2025. All rights reserved.
# This File is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License");
# Please refer to the Licence for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# makeself.cmake - 自定义 makeself 打包脚本

# 设置 makeself 路径
set(MAKESELF_EXE ${CPACK_CMAKE_BINARY_DIR}/makeself/makeself.sh)
set(MAKESELF_HEADER_EXE ${CPACK_CMAKE_BINARY_DIR}/makeself/makeself-header.sh)
if(NOT MAKESELF_EXE)
    message(FATAL_ERROR "makeself not found! Install it with: sudo apt install makeself")
endif()


# 创建临时安装目录
set(STAGING_DIR "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging")
file(MAKE_DIRECTORY "${STAGING_DIR}")

# 执行安装到临时目录
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${CPACK_CMAKE_BINARY_DIR}" --prefix "${STAGING_DIR}"
    RESULT_VARIABLE INSTALL_RESULT
)

if(NOT INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Installation to staging directory failed: ${INSTALL_RESULT}")
endif()

# 添加安装脚本
message(STATUS "CMAKE_SOURCE_DIR = ${CPACK_CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CPACK_CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR = ${CPACK_CMAKE_BINARY_DIR}")
message(STATUS "CPACK_PACKAGE_FILE_NAME = ${CPACK_PACKAGE_FILE_NAME}")
message(STATUS "STAGING_DIR = ${STAGING_DIR}")
message(STATUS "CMAKE_COMMAND = ${CMAKE_COMMAND}")
message(STATUS "CPACK_TEMPORARY_DIRECTORY = ${CPACK_TEMPORARY_DIRECTORY}")

file(STRINGS ${CPACK_CMAKE_BINARY_DIR}/makeself.txt script_output)
string(REPLACE " " ";" makeself_param_string "${script_output}")
string(REGEX MATCH "CANN.*\\.run" package_name "${makeself_param_string}")

message(STATUS "script output: ${script_output}")
message(STATUS "makeself: ${makeself_param_string}")
message(STATUS "package: ${package_name}")

execute_process(COMMAND bash ${MAKESELF_EXE}
    --header ${MAKESELF_HEADER_EXE}
    --help-header ops_transformer/script/help.info
    ${makeself_param_string} ops_transformer/script/install.sh
    WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
    RESULT_VARIABLE EXEC_RESULT
    ERROR_VARIABLE  EXEC_ERROR
)

if(NOT EXEC_RESULT EQUAL 0)
    message(FATAL_ERROR "makeself packaging failed: ${EXEC_ERROR}")
endif()

execute_process(COMMAND cp ${CPACK_EXTERNAL_BUILT_PACKAGES_DIR}/${package_name} ${CPACK_PACKAGE_DIRECTORY}/
    COMMAND echo "Copy ${CPACK_EXTERNAL_BUILT_PACKAGES_DIR}/${package_name} to ${CPACK_PACKAGE_DIRECTORY}/"
    WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
)
