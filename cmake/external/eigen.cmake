# Copyright(c) Huawei Technologies Co., Ltd.2025. All rights reserved.
# This File is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License");
# Please refer to the Licence for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================


set(_eigen_url "")
if(CANN_PKG_SERVER)
  set(_eigen_url "${CANN_PKG_SERVER}/libs/eigen3/eigen-3.4.0.tar.gz")
endif()

ExternalProject_Add(eigen
  URL               ${_eigen_url}
                    https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_MD5           4c527a9171d71a72a9d4186e65bea559
  DOWNLOAD_DIR      download/eigen
  PREFIX            third_party
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(eigen SOURCE_DIR)
ExternalProject_Get_Property(eigen BINARY_DIR)

set(EIGEN_INCLUDE ${SOURCE_DIR})

add_custom_target(eigen_headers ALL DEPENDS eigen)
