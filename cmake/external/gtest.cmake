# Copyright(c) Huawei Technologies Co., Ltd.2025. All rights reserved.
# This File is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License");
# Please refer to the Licence for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================


set(_gtest_url "")
if(CANN_PKG_SERVER)
  set(_gtest_url "${CANN_PKG_SERVER}/libs/ge_gtest/release-1.8.0.tar.gz")
endif()

ExternalProject_Add(external_gtest
  URL               ${_gtest_url}
                    https://github.com/google/googletest/archive/release-1.8.0.tar.gz
  URL_MD5           16877098823401d1bf2ed7891d7dce36
  DOWNLOAD_DIR      download/gtest
  PREFIX            third_party
  CMAKE_CACHE_ARGS
      -DBUILD_TESTING:BOOL=OFF
      -DBUILD_SHARED_LIBS:BOOL=ON
      -Dgtest_build_samples:BOOL=OFF
      -DCMAKE_C_FLAGS:STRING=-D_GLIBCXX_USE_CXX11_ABI=0
      -DCMAKE_CXX_FLAGS:STRING=-D_GLIBCXX_USE_CXX11_ABI=0
  INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(external_gtest SOURCE_DIR)
ExternalProject_Get_Property(external_gtest BINARY_DIR)

set(GTEST_INCLUDE ${SOURCE_DIR}/googletest/include)
set(GMOCK_INCLUDE ${SOURCE_DIR}/googlemock/include)

add_library(gtest SHARED IMPORTED)
add_dependencies(gtest external_gtest)
set_target_properties(gtest PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/googlemock/gtest/libgtest.so)

add_library(gtest_main SHARED IMPORTED)
add_dependencies(gtest_main external_gtest)
set_target_properties(gtest_main PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/googlemock/gtest/libgtest_main.so)
