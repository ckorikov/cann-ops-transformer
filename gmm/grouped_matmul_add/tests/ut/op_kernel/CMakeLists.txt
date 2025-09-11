if(UT_TEST_ALL OR FAST_OP_UT)
include_directories(
        ${ASCEND_DIR}/opensdk/opensdk/gtest/include
        ${ASCEND_DIR}/compiler/tikcpp/tikcfw
        ${ASCEND_DIR}/compiler/tikcpp/tikcfw/impl
        ${ASCEND_DIR}/compiler/tikcpp/tikcfw/interface
        ${CANN_ROOT}/ops/built-in/tbe/impl/ascendc/common/
        ${CANN_ROOT}/ops/built-in/tests/ut/op_tiling_test
        ${ASCEND_DIR}/runtime/highlevel_api
        ${ASCEND_DIR}/include/ascendc/highlevel_api
        ${CMAKE_CURRENT_SOURCE_DIR}/
        ${CANN_ROOT}/ops/common
        ${CANN_ROOT}/ops/built-in/tests/ut/fast_op_test
        )

SET(PRIVATE_L
    c_sec
    mmpa
    error_manager
    pem_davinci
    tikicpulib::ascend910B1
    pthread
)

add_library(grouped_matmul_add SHARED
        ${CANN_ROOT}/ops/transformer/grouped_matmul_add/op_kernel/grouped_matmul_add.cpp
            ./test_grouped_matmul_add.cpp
)

target_compile_options(grouped_matmul_add PRIVATE 
    -DDTYPE_X1=half -DDTYPE_X2=half -DDTYPE_BIAS=half -DDTYPE_Y=half
    -g -include ${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_add_tiling_def.h)

target_link_libraries(grouped_matmul_add PRIVATE
                      $<BUILD_INTERFACE:intf_llt_pub_asan>
                      -Wl,--whole-archive
                      -Wl,--no-as-needed
                      ${PRIVATE_L}
                      -Wl,--as-needed
                      -Wl,--no-whole-archive
)
endif()