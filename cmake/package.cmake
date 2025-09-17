# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#### CPACK to package run #####

# download makeself package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/external/makeself-fetch.cmake)

function(pack_custom)
  message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      message(STATUS "Detected architecture: x86_64")
      set(ARCH x86_64)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
      message(STATUS "Detected architecture: ARM64")
      set(ARCH aarch64)
  else ()
      message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif ()
  set(PACK_CUSTOM_NAME "cann-ops-transformer-${VENDOR_NAME}-linux-${ARCH}")
  npu_op_package(${PACK_CUSTOM_NAME}
    TYPE RUN
    CONFIG
      ENABLE_SOURCE_PACKAGE True
      ENABLE_BINARY_PACKAGE True
      INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/
      ENABLE_DEFAULT_PACKAGE_NAME_RULE False
  )

  npu_op_package_add(${PACK_CUSTOM_NAME}
    LIBRARY
      cust_opapi
  )
  if (TARGET cust_proto)
    npu_op_package_add(${PACK_CUSTOM_NAME}
        LIBRARY
        cust_proto
    )
  endif()
  if (TARGET cust_opmaster)
    npu_op_package_add(${PACK_CUSTOM_NAME}
        LIBRARY
        cust_opmaster
    )
  endif()
endfunction()

function(pack_built_in)
  #### built-in package ####
  message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      message(STATUS "Detected architecture: x86_64")
      set(ARCH x86_64)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
      message(STATUS "Detected architecture: ARM64")
      set(ARCH aarch64)
  else ()
      message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif ()

  set(script_prefix ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts)
  install(DIRECTORY ${script_prefix}/
      DESTINATION ops_transformer/script
      FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
      DIRECTORY_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )

  set(SCRIPTS_FILES
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.bash
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.csh
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.fish
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
  )

  install(FILES ${SCRIPTS_FILES}
      DESTINATION ops_transformer/script
  )
  set(COMMON_FILES
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/install_common_parser.sh
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func_v2.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_installer.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/script_operator.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_cfg.inc
  )

  set(PACKAGE_FILES
      ${COMMON_FILES}
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/multi_version.inc
  )
  set(LATEST_MANGER_FILES
      ${COMMON_FILES}
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
  )
  set(CONF_FILES
      ${CMAKE_SOURCE_DIR}/scripts/package/common/cfg/path.cfg
  )
  install(FILES ${CMAKE_SOURCE_DIR}/version.info
      DESTINATION .
  )
  install(FILES ${CONF_FILES}
      DESTINATION ops_transformer/conf
  )
  install(FILES ${PACKAGE_FILES}
      DESTINATION ops_transformer/script
  )
  install(FILES ${LATEST_MANGER_FILES}
      DESTINATION latest_manager
  )
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/package/latest_manager/scripts/
      DESTINATION latest_manager
  )
  set(BIN_FILES
      ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/prereq_check.bash
      ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/prereq_check.csh
      ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/prereq_check.fish
      ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/setenv.bash
      ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/setenv.csh
      ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/setenv.fish
  )
  install(FILES ${BIN_FILES}
      DESTINATION ops_transformer/bin
  )


  # ============= CPack =============
  set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
  set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")

  set(CPACK_INSTALL_PREFIX "/")

  set(CPACK_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
  set(CPACK_CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}")
  set(CPACK_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(CPACK_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  # set(CPACK_COMPONENTS_ALL runtime documentation)
  set(CPACK_ARCH "${ARCH}")
  set(CPACK_SET_DESTDIR ON)
  set(CPACK_GENERATOR External)
  set(CPACK_EXTERNAL_PACKAGE_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/makeself_built_in.cmake")
  set(CPACK_EXTERNAL_ENABLE_STAGING true)
  set(CPACK_PACKAGE_DIRECTORY "${CMAKE_INSTALL_PREFIX}")

  message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
  include(CPack)
endfunction()