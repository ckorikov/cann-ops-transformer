# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#/

if(TILING_UT OR (UT_TEST_ALL AND NOT AICPU_ONLY))
    message(STATUS "=== Debug: allto_allv_grouped_mat_mul ops_cpp_op_tiling_utest")
    target_sources(ops_cpp_op_tiling_utest PRIVATE
                   test_allto_allv_grouped_mat_mul_tiling.cpp
                   arch35/test_allto_allv_grouped_mat_mul_tiling_a5.cpp)
endif()

if(PROTO_UT OR (UT_TEST_ALL AND NOT AICPU_ONLY))
    message(STATUS "=== Debug: allto_allv_grouped_mat_mul ops_cpp_op_proto_utest")
    target_sources(ops_cpp_proto_utest PRIVATE
                   test_allto_allv_grouped_mat_mul_gentask_proto.cpp
                   test_allto_allv_grouped_mat_mul_runtime_proto.cpp)
endif()

if(OP_API_UT OR (UT_TEST_ALL AND NOT AICPU_ONLY))
    message(STATUS "=== Debug: allto_allv_grouped_mat_mul ops_cpp_op_api_utest")
    target_sources(op_api_l2_ut PRIVATE 
                   test_allto_allv_grouped_mat_mul_api.cpp)
    target_include_directories(op_api_l2_ut PRIVATE
                               ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/op_api)
endif()
