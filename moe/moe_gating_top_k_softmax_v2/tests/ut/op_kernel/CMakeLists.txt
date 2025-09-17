# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

file(GLOB CURRENT_SOURCE_DIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
message(STATUS "=== Debug: CURRENT_SOURCE_DIRS =${CURRENT_SOURCE_DIRS} ")
foreach(SUB_DIR ${CURRENT_SOURCE_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

# 算子自己的tiling文件路径
set(moe_gating_top_k_softmax_v2_tiling_files
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/moe_gating_top_k_softmax_v2_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/moe_gating_top_k_softmax_v2_tiling_perf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/moe_gating_top_k_softmax_v2_tiling_k_renorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/moe_gating_top_k_softmax_v2_tiling_k_fullload.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/moe_gating_top_k_softmax_v2_tiling_ek_fullload.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/moe_gating_top_k_softmax_v2_tiling_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/moe_gating_top_k_softmax_v2_regbase_tiling_perf.cpp
    )
# 使用AddOpTestCase
# param1：算子名称，以kernel方式命名
# param2：soc版本，多个以分号分隔，例如："ascend910_9599;AscendB1"
# param3：自定义编译选项，一般填写测试的一种典型数据类型组合，不需要则传入空字符串，例如："-DDTYPE_X=float"，多个使用空格分隔，例如："-DDTYPE_X=float -DDTYPE_Y=float"
# param4：该算子依赖的所有tiling源码文件
if (UT_TEST_ALL OR FAST_OP_UT)
    #AddOpTestCase(moe_gating_top_k_softmax_v2 "ascend910B1" "-DDTYPE_X=float" "${moe_gating_top_k_softmax_v2_tiling_files}")
endif()
