# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_executable(test_nsa_compress_with_cache
        test_nsa_compress_with_cache.cpp
)

add_execute_example(
        TARGET_NAME test_nsa_compress_with_cache
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_nsa_compress_with_cache_case.sh
        TEST_CASE test_nsa_compress_with_cache
        ACLNN_FUNC "aclnnNsaCompressWithCache"
)

target_include_directories(test_nsa_compress_with_cache PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_CUSTOM_OPP_PATH}/include
)

install(TARGETS test_nsa_compress_with_cache
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        OPTIONAL
)
