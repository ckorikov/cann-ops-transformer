# Copyright(c) Huawei Technologies Co., Ltd.2025. All rights reserved.
# This File is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License");
# Please refer to the Licence for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================


set(_json_url "")
if(CANN_PKG_SERVER)
  set(_json_url "${CANN_PKG_SERVER}/libs/json/v3.6.1/include.zip")
endif()

ExternalProject_Add(nlohmann_json
  URL               ${_json_url}
                    https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip
  URL_MD5           0dc903888211db3a0f170304cd9f3a89
  DOWNLOAD_DIR      download/nlohmann_json
  PREFIX            third_party
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(nlohmann_json SOURCE_DIR)
ExternalProject_Get_Property(nlohmann_json BINARY_DIR)

set(JSON_INCLUDE ${SOURCE_DIR})
add_library(json INTERFACE)
target_include_directories(json INTERFACE ${JSON_INCLUDE})
add_dependencies(json nlohmann_json)
