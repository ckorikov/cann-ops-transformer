# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
include(ExternalProject)

set(ASCEND_PROTOBUF_DIR ${CANN_3RD_LIB_PATH}/ascend_protobuf)
# 从已经下载好的路径找
find_path(ASCEND_PROTOBUF_SHARED_INCLUDE
    NAMES google/protobuf/api.pb.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

find_path(ASCEND_PROTOC
    NAMES protoc
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

set(ascend_protobuf_transformer_FOUND FALSE)
if(ASCEND_PROTOBUF_SHARED_INCLUDE AND ASCEND_PROTOC)
    set(ascend_protobuf_transformer_FOUND TRUE)
endif()

if(ascend_protobuf_transformer_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[ThirdPartyLib][ascend protobuf] ascend_protobuf_shared found, skip compile.")
    cmake_print_variables(ASCEND_PROTOBUF_SHARED_INCLUDE)
    cmake_print_variables(ASCEND_PROTOC)
    set(Protobuf_INCLUDE ${ASCEND_PROTOBUF_SHARED_INCLUDE})
    set(Protobuf_PATH ${ASCEND_PROTOC})
    set(Protobuf_PROTOC_EXECUTABLE ${Protobuf_PATH}/protoc)
else()
    message(STATUS "[ThirdPartyLib][ascend protobuf] ascend protobuf shared not found, finding binary file.")
    if(EXISTS ${OPEN_SOURCE_DIR})
        message(STATUS "OPEN_SOURCE_DIR exist, OPEN_SOURCE_DIR is ${OPEN_SOURCE_DIR}")
    elseif(EXISTS ${OPEN_SOURCE_DIR_BAK})
        message(STATUS "OPEN_SOURCE_DIR_BAK exist, OPEN_SOURCE_DIR_BAK is ${OPEN_SOURCE_DIR_BAK}")
        set(OPEN_SOURCE_DIR ${OPEN_SOURCE_DIR_BAK})
    endif()

    set(REQ_URL "${CANN_3RD_LIB_PATH}/protobuf/protobuf-all-25.1.tar.gz")
    set(REQ_URL_BACK "${CANN_3RD_LIB_PATH}/protobuf/protobuf-25.1.tar.gz")
    set(REQ_URL_TMP "${OPEN_SOURCE_DIR}/protobuf/protobuf-all-25.1.tar.gz")
    set(REQ_URL_TMP_BACK "${OPEN_SOURCE_DIR}/protobuf/protobuf-25.1.tar.gz")
    # 初始化可选参数列表
    if(EXISTS ${REQ_URL})
        message(STATUS "[ThirdPartyLib][ascend protobuf] ${REQ_URL} found, start compile.")
    elseif(EXISTS ${REQ_URL_BACK})
        message(STATUS "[ThirdPartyLib][ascend protobuf] ${REQ_URL_BACK} found, start compile.")
        set(REQ_URL ${REQ_URL_BACK})
    elseif(EXISTS ${REQ_URL_TMP})
        message(STATUS "[ThirdPartyLib][ascend protobuf] ${REQ_URL_TMP} found, start compile.")
        set(REQ_URL ${REQ_URL_TMP})
    elseif(EXISTS ${REQ_URL_TMP_BACK})
        message(STATUS "[ThirdPartyLib][ascend protobuf] ${REQ_URL_TMP_BACK} found, start compile.")
        set(REQ_URL ${REQ_URL_TMP_BACK})
    else()
        message(STATUS "[ThirdPartyLib][ascend protobuf] ${REQ_URL} not found, need download.")
        set(REQ_URL "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz")
    endif()
    
    set(protobuf_CXXFLAGS "-Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -Dgoogle=ascend_private")
    set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")

    ExternalProject_Add(ascend_protobuf_build_transformer
                        URL ${REQ_URL}
                        DOWNLOAD_DIR download/ascend_protobuf
                        PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/build/modules/patch/protobuf_25.1_change_version.patch
                        CONFIGURE_COMMAND ${CMAKE_COMMAND}
                            -DCMAKE_INSTALL_LIBDIR=lib
                            -Dprotobuf_WITH_ZLIB=OFF
                            -DLIB_PREFIX=ascend_
                            -DCMAKE_SKIP_RPATH=TRUE
                            -Dprotobuf_BUILD_TESTS=OFF
                            -DBUILD_SHARED_LIBS=OFF
                            -DCMAKE_CXX_STANDARD=14
                            -DCMAKE_CXX_FLAGS=${protobuf_CXXFLAGS}
                            -DCMAKE_CXX_LDFLAGS=${protobuf_LDFLAGS}
                            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                            -DCMAKE_INSTALL_PREFIX=${ASCEND_PROTOBUF_DIR}
                            -Dprotobuf_BUILD_PROTOC_BINARIES=ON
                            -Dprotobuf_ABSL_PROVIDER=module
                            -DABSL_ROOT_DIR=${CMAKE_BINARY_DIR}/abseil_build_transformer-prefix/src/abseil_build_transformer
                            <SOURCE_DIR>
                        BUILD_COMMAND $(MAKE)
                        INSTALL_COMMAND ""
                        EXCLUDE_FROM_ALL TRUE
    )
    add_dependencies(ascend_protobuf_build_transformer abseil_build_transformer)

    ExternalProject_Get_Property(ascend_protobuf_build_transformer SOURCE_DIR)
    ExternalProject_Get_Property(ascend_protobuf_build_transformer BINARY_DIR)

    set(Protobuf_INCLUDE ${SOURCE_DIR}/src)
    set(Protobuf_PATH ${BINARY_DIR})
    set(Protobuf_PROTOC_EXECUTABLE ${Protobuf_PATH}/protoc)

    add_custom_command(
        OUTPUT ${Protobuf_PROTOC_EXECUTABLE}
        DEPENDS ascend_protobuf_build_transformer
    )
endif()