# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

unset(json_FOUND CACHE)
unset(JSON_INCLUDE CACHE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(json
        FOUND_VAR
        json_FOUND
        REQUIRED_VARS
        JSON_INCLUDE)

if(json_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "json found in ${JSON_INCLUDE}, and not force rebuild cann third_party")
    set(JSON_INCLUDE_DIR ${JSON_INCLUDE})
else()
    if(EXISTS ${OPEN_SOURCE_DIR})
        message(STATUS "OPEN_SOURCE_DIR exist, OPEN_SOURCE_DIR is ${OPEN_SOURCE_DIR}")
    elseif(EXISTS ${OPEN_SOURCE_DIR_BAK})
        message(STATUS "OPEN_SOURCE_DIR_BAK exist, OPEN_SOURCE_DIR_BAK is ${OPEN_SOURCE_DIR_BAK}")
        set(OPEN_SOURCE_DIR ${OPEN_SOURCE_DIR_BAK})
    endif()

    if (EXISTS "${CANN_3RD_LIB_PATH}/json/include.zip")
        set(REQ_URL "${CANN_3RD_LIB_PATH}/json/include.zip")
    elseif (IS_DIRECTORY "${CANN_3RD_LIB_PATH}/json")
        set(REQ_URL "${CANN_3RD_LIB_PATH}/json")
    elseif (EXISTS "${OPEN_SOURCE_DIR}/json/include.zip")
        set(REQ_URL "${OPEN_SOURCE_DIR}/json/include.zip")
    elseif (IS_DIRECTORY "${OPEN_SOURCE_DIR}/json")
        set(REQ_URL "${OPEN_SOURCE_DIR}/json")
    else()
        set(REQ_URL "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip")
    endif()

    message(STATUS "json not found, REQ_URL is ${REQ_URL}")

    include(ExternalProject)
    ExternalProject_Add(third_party_json
        URL               ${REQ_URL}
        DOWNLOAD_DIR      download/json
        PREFIX            third_party
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
    )

    ExternalProject_Get_Property(third_party_json SOURCE_DIR)
    set(JSON_INCLUDE_DIR ${SOURCE_DIR}/include)
    add_library(json INTERFACE)
    add_dependencies(json third_party_json)
endif()
