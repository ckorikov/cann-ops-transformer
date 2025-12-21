# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

include(ExternalProject)

if(EXISTS ${OPEN_SOURCE_DIR})
  message(STATUS "OPEN_SOURCE_DIR exist, OPEN_SOURCE_DIR is ${OPEN_SOURCE_DIR}")
elseif(EXISTS ${OPEN_SOURCE_DIR_BAK})
  message(STATUS "OPEN_SOURCE_DIR_BAK exist, OPEN_SOURCE_DIR_BAK is ${OPEN_SOURCE_DIR_BAK}")
  set(OPEN_SOURCE_DIR ${OPEN_SOURCE_DIR_BAK})
endif()

set(REQ_URL "${CANN_3RD_LIB_PATH}/abseil-cpp/abseil-cpp-20230802.1.tar.gz")
set(ABSEIL_URL "${OPEN_SOURCE_DIR}/abseil-cpp/abseil-cpp-20230802.1.tar.gz")
message(STATUS "CANN_3RD_LIB_PATH is ${CANN_3RD_LIB_PATH}")
message(STATUS "OPEN_SOURCE_DIR is ${OPEN_SOURCE_DIR}")
# 初始化可选参数列表
if(EXISTS ${REQ_URL})
  message(STATUS "[ThirdPartyLib][abseil-cpp] ${REQ_URL} found.")
elseif(EXISTS ${ABSEIL_URL})
  message(STATUS "[ThirdPartyLib][abseil-cpp] ${ABSEIL_URL} found.")
  set(REQ_URL ${ABSEIL_URL})
else()
  message(STATUS "[ThirdPartyLib][abseil-cpp] ${REQ_URL} not found, need download.")
  set(REQ_URL "https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz")
endif()

ExternalProject_Add(abseil_build_transformer
                    URL ${REQ_URL}
                    DOWNLOAD_DIR download/abseil_transformer
                    PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/build/modules/patch/protobuf-hide_absl_symbols.patch
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    EXCLUDE_FROM_ALL TRUE 
)

ExternalProject_Get_Property(abseil_build_transformer SOURCE_DIR)
set(ABSL_SOURCE_DIR ${SOURCE_DIR})