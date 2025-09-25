# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

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
