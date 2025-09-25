# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_executable(test_mla_prolog_nq_bsh
        test_mla_prolog_nq_bsh.cpp
)
add_execute_example(
        TARGET_NAME test_mla_prolog_nq_bsh
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_mla_prolog_case.sh
        TEST_CASE mla_prolog_nq_bsh_case
        ACLNN_FUNC "aclnnMlaProlog"
)

add_executable(test_mla_prolog_nq_tnd
        test_mla_prolog_nq_tnd.cpp
)
add_execute_example(
        TARGET_NAME test_mla_prolog_nq_tnd
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_mla_prolog_case.sh
        TEST_CASE mla_prolog_nq_tnd_case
        ACLNN_FUNC "aclnnMlaProlog"
)

add_executable(test_mla_prolog_pqkvnq_bsh
        test_mla_prolog_pqkvnq_bsh.cpp
)
add_execute_example(
        TARGET_NAME test_mla_prolog_pqkvnq_bsh
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_mla_prolog_case.sh
        TEST_CASE mla_prolog_pqkvnq_bsh_case
        ACLNN_FUNC "aclnnMlaProlog"
)

add_executable(test_mla_prolog_pqkvnq_tnd
        test_mla_prolog_pqkvnq_tnd.cpp
)
add_execute_example(
        TARGET_NAME test_mla_prolog_pqkvnq_tnd
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_mla_prolog_case.sh
        TEST_CASE mla_prolog_pqkvnq_tnd_case
        ACLNN_FUNC "aclnnMlaProlog"
)

add_executable(test_mla_prolog_pqkvq_bsh
        test_mla_prolog_pqkvq_bsh.cpp
)
add_execute_example(
        TARGET_NAME test_mla_prolog_pqkvq_bsh
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_mla_prolog_case.sh
        TEST_CASE mla_prolog_pqkvq_bsh_case
        ACLNN_FUNC "aclnnMlaProlog"
)

add_executable(test_mla_prolog_pqkvq_tnd
        test_mla_prolog_pqkvq_tnd.cpp
)
add_execute_example(
        TARGET_NAME test_mla_prolog_pqkvq_tnd
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_mla_prolog_case.sh
        TEST_CASE mla_prolog_pqkvq_tnd_case
        ACLNN_FUNC "aclnnMlaProlog"
)

install(TARGETS test_mla_prolog_nq_bsh
        test_mla_prolog_nq_tnd
        test_mla_prolog_pqkvnq_bsh
        test_mla_prolog_pqkvnq_tnd
        test_mla_prolog_pqkvq_bsh
        test_mla_prolog_pqkvq_tnd
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        OPTIONAL
)

target_link_libraries(test_mla_prolog_nq_bsh PRIVATE
    -lc_sec
)

target_link_libraries(test_mla_prolog_nq_tnd PRIVATE
    -lc_sec
)

target_link_libraries(test_mla_prolog_pqkvnq_bsh PRIVATE
    -lc_sec
)

target_link_libraries(test_mla_prolog_pqkvnq_tnd PRIVATE
    -lc_sec
)

target_link_libraries(test_mla_prolog_pqkvq_bsh PRIVATE
    -lc_sec
)

target_link_libraries(test_mla_prolog_pqkvq_tnd PRIVATE
    -lc_sec
)

