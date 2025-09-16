#
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

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        rope_quant_kvcache_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    add_ops_compile_options(
        OP_NAME RopeQuantKvcache
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_modules_sources(OPTYPE rope_quant_kvcache ACLNNTYPE aclnn)
endif()