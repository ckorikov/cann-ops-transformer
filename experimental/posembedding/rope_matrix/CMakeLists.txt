cmake_minimum_required(VERSION 3.16.0)
project(Ascend_C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")

set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_TOOLKIT_HOME})
set(SOC_VERSION "Ascend910B3")
set(RUN_MODE "npu")

include(${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(rope_matrix_custom_kernel SHARED
    op_device/rope_matrix.cpp op_device/rope_matrix.h op_device/rope_matrix_cube.h inc/rope_matrix_tiling.h
)

ascendc_compile_definitions(rope_matrix_custom_kernel PRIVATE
  -DHAVE_WORKSPACE
  -DHAVE_TILING
)