# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
if (BUILD_TORCH_OPS)
    set(OPERATOR_NAME "groupedmatmul")
    message(STATUS "BUILD_TORCH_OPS ON in ${OPERATOR_NAME}")
    
    set(OPERATOR_TARGET "${OPERATOR_NAME}_objects")
    set(OPERATOR_CONFIG "${OPERATOR_NAME}:${OPERATOR_TARGET}" PARENT_SCOPE)

    file(GLOB OPERATOR_MAIN_SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        EXCLUDE 
        "${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_kernel_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_kernel_*.cpp"
    )

    # Mark .cpp files with special properties
    set_source_files_properties(
        ${OPERATOR_MAIN_SOURCES} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
    )

    add_library(${OPERATOR_TARGET} OBJECT ${OPERATOR_MAIN_SOURCES})

    target_compile_options(${OPERATOR_TARGET} PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_include_directories(${OPERATOR_TARGET} PRIVATE ${COMMON_INCLUDE_DIRS})

    # # 2. 数据类型常量定义（必须与 grouped_matmul_utils.h 中的 DT_* 常量一致）
    # set(DTYPE_DEFS
    #     -DDT_FLOAT=0           
    #     -DDT_FLOAT16=1         
    #     -DDT_INT8=2                     
    #     -DDT_INT32=3                      
    #     -DDT_BF16=27         
    #     -DDT_INT4=29           
    # )

    set(COMBO_DEFS_IDX_0
        -DORIG_DTYPE_X=DT_FLOAT16
        -DDTYPE_X=half

        -DORIG_DTYPE_WEIGHT=DT_FLOAT16
        -DDTYPE_WEIGHT=half

        -DORIG_DTYPE_Y=DT_FLOAT16
        -DDTYPE_Y=half

        -DDTYPE_BIAS=half
        -DDTYPE_SCALE=uint64_t
        -DDTYPE_ANTIQUANT_SCALE=half
        -DCOMBO_INDEX=0
    )

    set(COMBO_DEFS_IDX_1
        -DORIG_DTYPE_X=DT_BF16
        -DDTYPE_X=bfloat16_t

        -DORIG_DTYPE_WEIGHT=DT_BF16
        -DDTYPE_WEIGHT=bfloat16_t

        -DORIG_DTYPE_Y=DT_BF16
        -DDTYPE_Y=half

        -DDTYPE_BIAS=float
        -DDTYPE_SCALE=uint64_t
        -DDTYPE_ANTIQUANT_SCALE=half
        -DCOMBO_INDEX=1
    )

     set(NPU_KERNEL_COMPILE_FLAGS_STR
        "--cce-soc-version=Ascend910B1"
        "--cce-soc-core-type=VecCore"
        "--cce-auto-sync"
        "-xcce"
        "-D__CCE_KT_TEST__"
        "-D__AI_CORE__"
        "-D__AI_AICPU__"
        "-D__CCE_AICORE__=220"  # 合并到编译标志，避免重复设置
    )

    # 为每个下标编译独立的kernel目标文件
    foreach(INDEX RANGE 0 1)
        set(KERNEL_TARGET "${OPERATOR_NAME}_kernel_${INDEX}")
        set(KERNEL_SOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_kernel_${INDEX}.cpp")
        
        # 确保文件存在（修复空文件问题）
        if(NOT EXISTS ${KERNEL_SOURCE_FILE})
            # 写入有效内容：包含impl + 声明COMBO_INDEX（避免编译空文件）
            file(WRITE ${KERNEL_SOURCE_FILE} 
                "#define COMBO_INDEX ${INDEX}\n"
                "#include \"grouped_matmul_kernel_impl.cpp\"\n"
            )
        endif()
        
        add_library(${KERNEL_TARGET} OBJECT ${KERNEL_SOURCE_FILE})
        
        # 3. 修复：通过 target_compile_options 传递编译标志（推荐方式）
        target_compile_options(${KERNEL_TARGET} PRIVATE
            ${COMMON_COMPILE_OPTIONS}  # 复用公共编译选项
            ${NPU_KERNEL_COMPILE_FLAGS_STR}  # 昇腾平台编译标志
        )
        
        # 设置编译宏（包含DTYPE_DEFS和COMBO_DEFS）
        target_compile_definitions(${KERNEL_TARGET} PRIVATE
            ${DTYPE_DEFS}
            ${COMBO_DEFS_IDX_${INDEX}}
            -D__CCE_AICORE__  # 平台宏
        )
        
        target_include_directories(${KERNEL_TARGET} PRIVATE
            ${COMMON_INCLUDE_DIRS}
            "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        
        target_sources(${OPERATOR_TARGET} PRIVATE 
            $<TARGET_OBJECTS:${KERNEL_TARGET}>
        )
        
        message(STATUS "Added kernel target for combo index ${INDEX}: ${KERNEL_TARGET}")
    endforeach()

    return()
endif()