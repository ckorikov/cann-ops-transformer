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

set(_NCWC_OpApiSourcesExt
        ${OPS_ADV_DIR}/src/transformer/nsa_compress_with_cache/ophost/nsa_compress_with_cache.cpp
        ${OPS_ADV_DIR}/src/transformer/nsa_compress_with_cache/ophost/aclnn_nsa_compress_with_cache.cpp
)

set(_NCWC_OpProtoSourcesExt
        ${OPS_ADV_DIR}/src/transformer/nsa_compress_with_cache/ophost/nsa_compress_with_cache_proto.cpp
)

set(_NCWC_OpTilingSourcesExt)
file(GLOB _Src1 "${OPS_ADV_DIR}/src/transformer/nsa_compress_with_cache/ophost/nsa_compress_with_cache_tiling*.cc")
file(GLOB _Src2 "${OPS_ADV_DIR}/src/transformer/nsa_compress_with_cache/ophost/nsa_compress_with_cache_tiling*.cpp")
list(APPEND _NCWC_OpTilingSourcesExt ${_Src1} ${_Src2})

set(_NCWC_OpTilingPrivateIncludesExt)

set(_NCWC_OpKernelTilingDataDefH
        # 公共
        ${OPS_ADV_DIR}/src/utils/inc/tiling/data_copy_transpose_tiling_def.h
        # NCWC
        ${OPS_ADV_DIR}/src/transformer/nsa_compress_with_cache/ophost/nsa_compress_with_cache_tiling.h
)

set(_NCWC_OpKernelSourcesExt)

set(_NCWC_OpKernelPrivateIncludesExt
        ${OPS_ADV_DIR}/src/utils/inc/kernel
)

set(_NCWC_OpKernelPrivateCompileDefinitionsExt
        KernelCtrlParam nsa_compress_with_cache fp16 ORIG_DTYPE_QUERY=DT_FLOAT16 DTYPE_DQ=half       KFC_L1_RESERVER_SIZE=0
        KernelCtrlParam nsa_compress_with_cache bf16 ORIG_DTYPE_QUERY=DT_BF16    DTYPE_DQ=bfloat16_t KFC_L1_RESERVER_SIZE=0
)
set(_NCWC_UTestCommonPrivateIncludeExt
        ${OPS_ADV_DIR}/src/transformer/nsa_compress_with_cache/ophost
)
set(_NCWC_UTestCommonPrivateLinkLibrariesExt
        ops_utils_tiling_headers
        ${UTest_NamePrefix}_OpTiling
        error_manager
)

set(_NCWC_UTestSourcesExtForce)
file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cpp")
list(APPEND _NCWC_UTestSourcesExtForce ${_Src1} ${_Src2})

set(_NCWC_UTestAclnnSourcesExtForce)
file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cpp")
list(APPEND _NCWC_UTestAclnnSourcesExtForce ${_Src1} ${_Src2})

set(_NCWC_UTestCompileDefinitions)

list(APPEND _NCWC_OpKernelSourcesExt
        ${OPS_ADV_DIR}/src/transformer/nsa_compress_with_cache/nsa_compress_with_cache.cpp
)

file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cpp")
list(APPEND _NCWC_UTestSourcesExtForce ${_Src1} ${_Src2})

file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/*.cpp")
list(APPEND _NCWC_UTestAclnnSourcesExtForce ${_Src1} ${_Src2})

list(APPEND _NCWC_UTestCompileDefinitions)


OpsTest_Level2_AddOp(
        SUB_SYSTEM                                transformer
        BRIEF                                     NCWC
        SNAKE                                     nsa_compress_with_cache
        OPAPI_SOURCES_EXT                         ${_NCWC_OpApiSourcesExt}
        PROTO_SOURCES_EXT                         ${_NCWC_OpProtoSourcesExt}
        TILING_SOURCES_EXT                        ${_NCWC_OpTilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT               ${_NCWC_OpTilingPrivateIncludesExt}
        KERNEL_SOURCES_EXT                        ${_NCWC_OpKernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                  ${_NCWC_OpKernelTilingDataDefH}
        KERNEL_PRIVATE_INCLUDES_EXT               ${_NCWC_OpKernelPrivateIncludesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT    ${_NCWC_OpKernelPrivateCompileDefinitionsExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT         ${_NCWC_UTestCommonPrivateIncludeExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT   ${_NCWC_UTestCommonPrivateLinkLibrariesExt}
        UTEST_COMMON_PRIVATE_COMPILE_DEFINITIONS  ${_NCWC_UTestCompileDefinitions}
        UTEST_SOURCES_EXT                         ${_NCWC_UTestSourcesExtForce}
        UTEST_SOURCES_EXT_FORCE                   ON
        UTEST_PRIVATE_COMPILE_DEFINITIONS         ${_NCWC_UTestCompileDefinitions}
        UTEST_ACLNN_SOURCES_EXT                   ${_NCWC_UTestAclnnSourcesExtForce}
        UTEST_ACLNN_SOURCES_EXT_FORCE             ON
        UTEST_ACLNN_PRIVATE_COMPILE_DEFINITIONS   ${_NCWC_UTestCompileDefinitions}
)
