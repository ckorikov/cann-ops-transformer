# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations")

set(_ncai_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/nsa_compress_attention_infer/ophost/nsa_compress_attention_infer_tiling.cpp
)

set(_ncai_KernelTilingDataDefH
        ${OPS_ADV_DIR}/src/utils/inc/tiling/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/src/transformer/nsa_compress_attention_infer/ophost/nsa_compress_attention_infer_tiling.h
)

set(_ncai_CompileDefintions
        ORIG_DTYPE_QUERY=DT_FLOAT16
)

set(_ncai_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/src/transformer/nsa_compress_attention_infer/ophost/
)

set(_ncai_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
        ${UTest_NamePrefix}_OpTiling
        error_manager
)

set(_ncai_OpKernelPrivateIncludesExt
        ${OPS_ADV_DIR}/src/utils/inc/kernel
)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/nsa_compress_attention_infer _ncai_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   ncaInfer
        SNAKE                                   nsa_compress_attention_infer
        KERNEL_TILING_DATA_DEF_H                ${_ncai_KernelTilingDataDefH}
        KERNEL_PRIVATE_INCLUDES_EXT             ${_ncai_OpKernelPrivateIncludesExt}
        KERNEL_SOURCES_EXT                      ${_ncai_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_ncai_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_ncai_KernelTilingExtInclude}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_ncai_CompileDefintions}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_ncai_TargetPrivateLinkLibrariesExt}
)
