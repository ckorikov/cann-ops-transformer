# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR ONNX_PLUGIN_UT)
  ## set variables
  set(ONNX_PLUGIN_UT_EXE ${PKG_NAME}_onnx_plugin_ut)

  ## add onnx_plugin ut cases static lib: libmath_op_onnx_plugin_ut_cases.a
  add_onnx_plugin_ut_modules(${ONNX_PLUGIN_MODULE_NAME})

  ## add onnx_plugin source so: libophost_math_ut.so
  add_library(${ONNX_PLUGIN_NAME}_ut SHARED
      $<$<TARGET_EXISTS:${ONNX_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${ONNX_PLUGIN_NAME}_obj>>
  )
  target_link_libraries(${ONNX_PLUGIN_NAME}_ut PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      -Wl,--no-as-needed
          $<$<TARGET_EXISTS:opsbase>:opsbase>
      -Wl,--as-needed
      -Wl,--whole-archive
          rt2_registry_static
      -Wl,--no-whole-archive
  )

  ## add ophost ut exe:  math_op_host_ut
  add_executable(${ONNX_PLUGIN_UT_EXE}
      test_op_onnx_plugin_main.cpp
  )
  target_compile_definitions(${ONNX_PLUGIN_UT_EXE} PRIVATE)
  target_compile_options(${ONNX_PLUGIN_UT_EXE} PUBLIC -fPIE -fno-access-control)
  target_include_directories(${ONNX_PLUGIN_UT_EXE} PRIVATE
      ${ASCEND_DIR}/pkg_inc
      )
  target_link_directories(${ONNX_PLUGIN_UT_EXE} PRIVATE
      ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
      )
  target_link_libraries(${ONNX_PLUGIN_UT_EXE} PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      -Wl,--whole-archive
          ${ONNX_PLUGIN_MODULE_NAME}_cases
          fmk_onnx_parser
          parser_common
          ${ONNX_PLUGIN_NAME}_ut
      -Wl,--no-whole-archive
          opp_registry
      -Wl,--no-as-needed
          metadef
          unified_dlog
      -Wl,--as-needed
          register
          graph
          platform
          gtest
  )

  ## exec ophost ut exe
  if(ENABLE_UT_EXEC)
    if(ENABLE_ASAN)
      execute_process(
        COMMAND ${CMAKE_C_COMPILER} -print-file-name=libasan.so
        OUTPUT_VARIABLE LIBASAN_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE result
      )
      if(NOT result EQUAL 0)
        message(FATAL_ERROR "compiler not support asan, please disable asan")
      endif()

      execute_process(
        COMMAND ${CMAKE_C_COMPILER} -print-file-name=libstdc++.so
        OUTPUT_VARIABLE LIBSTDC_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE stdc_result
      )
      if(NOT stdc_result EQUAL 0)
        message(FATAL_ERROR "not find libstdc++.so, please confirm.")
      endif()
      
      add_custom_command(
        TARGET ${ONNX_PLUGIN_UT_EXE}
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
          LD_PRELOAD=${LIBASAN_PATH}:${LIBSTDC_PATH}
          ASAN_OPTIONS=detect_leaks=0 $<TARGET_FILE:${ONNX_PLUGIN_UT_EXE}>
        COMMENT "Run ops op_host utest with asan"
        )
    else()
      add_custom_command(
        TARGET ${ONNX_PLUGIN_UT_EXE}
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} $<TARGET_FILE:${ONNX_PLUGIN_UT_EXE}>
        COMMENT "Run ops op_host utest"
        )
    endif()
  endif()

  if(ENABLE_VALGRIND)
    add_custom_command(
      TARGET ${ONNX_PLUGIN_UT_EXE}
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} && valgrind --undef-value-errors=no
        --leak-check=full $<TARGET_FILE:${ONNX_PLUGIN_UT_EXE}>
      COMMENT "Run ops op_host utest by valgrind"
      )
  endif()
endif()
