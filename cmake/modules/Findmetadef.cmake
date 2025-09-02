# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(metadef_FOUND)
  message(STATUS "metadef has been found")
  return()
endif()

include(FindPackageHandleStandardArgs)

set(METADEF_HEAD_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
  ${TOP_DIR}/metadef/inc/external/            # compile with ci
)

set(METADEF_LIB_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}
)

find_path(METADEF_INC_DIR
  NAMES register/register.h
  PATHS ${METADEF_HEAD_SEARCH_PATHS}
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

find_library(REGISTER_LIB_DIR
  NAME register
  PATHS ${METADEF_LIB_SEARCH_PATHS}
  PATH_SUFFIXES lib64
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(REGISTER_LIB_DIR)
  get_filename_component(REGISTER_LIB_DIR ${REGISTER_LIB_DIR} REALPATH)
  add_library(register SHARED IMPORTED)
  set_target_properties(register PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${METADEF_INC_DIR}
    IMPORTED_LOCATION ${REGISTER_LIB_DIR}
  )
  message(STATUS "Found register library:${REGISTER_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library register")
  endif()
endif()

find_library(EXEGRAPH_LIB_DIR
  NAME exe_graph
  PATHS ${METADEF_LIB_SEARCH_PATHS}
  PATH_SUFFIXES lib64
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(EXEGRAPH_LIB_DIR)
  get_filename_component(EXEGRAPH_LIB_DIR ${EXEGRAPH_LIB_DIR} REALPATH)
  add_library(exe_graph SHARED IMPORTED)
  set_target_properties(exe_graph PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${METADEF_INC_DIR}/exe_graph
    IMPORTED_LOCATION ${EXEGRAPH_LIB_DIR}
  )
  message(STATUS "Found exe_graph library:${EXEGRAPH_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library exe_graph")
  endif()
endif()

find_library(REGISTER_STATIC_LIB_DIR
  NAME librt2_registry.a
  PATHS ${METADEF_LIB_SEARCH_PATHS}
  PATH_SUFFIXES lib64
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(REGISTER_STATIC_LIB_DIR)
  get_filename_component(REGISTER_STATIC_LIB_DIR ${REGISTER_STATIC_LIB_DIR} REALPATH)
  add_library(rt2_registry_static STATIC IMPORTED)
  set_target_properties(rt2_registry_static PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${METADEF_INC_DIR}
    IMPORTED_LOCATION ${REGISTER_STATIC_LIB_DIR}
  )
  message(STATUS "Found rt2_registry library:${REGISTER_STATIC_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library rt2_registry")
  endif()
endif()

find_package_handle_standard_args(metadef
      REQUIRED_VARS METADEF_INC_DIR)

get_filename_component(METADEF_INC_DIR ${METADEF_INC_DIR} REALPATH)
if(metadef_FOUND)
  set(METADEF_INCLUDE_DIRS
    ${METADEF_INC_DIR}/
    ${METADEF_INC_DIR}/exe_graph
  )

  if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    set(METADEF_INCLUDE_DIRS ${METADEF_INC_DIR}/../ ${METADEF_INCLUDE_DIRS})
  endif()
  message(STATUS "Found source metadef include dir:  ${METADEF_INCLUDE_DIRS}")
endif()