# -*- coding: utf-8 -*-
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from ascendc_extension import AscendCExtension
CURRENT_DIR = os.path.dirname(__file__)

package_name = 'select_attn_decoding_ops'
version='0.3.1'
num_cores=os.environ.get('NUM_CORES', 0)

setup(
    name=package_name,
    version=version,
    ext_modules=[
        AscendCExtension(
            name=package_name,
            sources=['torch_interface.cpp'],
            extra_library_dirs=[os.path.join(CURRENT_DIR, 'lib')],  # location of custom lib{name}.so file
            extra_libraries=['quest_block_select', 'quest_block_select_paged'],  # names of custom lib{name}.so files
            extra_link_args=[
                '-L', os.path.join(CURRENT_DIR, 'lib'),  # Linker path to the shared library dir
                '-lquest_block_select',       # Shared library 1/2 name
                '-lquest_block_select_paged'  # Shared library 2/2 name
            ],
            runtime_library_dirs=[os.path.join(CURRENT_DIR, 'lib')],  # add the directory to RPATH
            extra_compile_args=[f'-DNUM_CORES={num_cores}']
            ),
        ],
    cmdclass={'build_ext': BuildExtension}
)
