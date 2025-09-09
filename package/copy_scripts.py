#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""复制脚本。"""

import os
import re
from datetime import datetime
from io import FileIO
from typing import Dict


STRIP_PATTERN = re.compile(r'# \[rm\-before\-pkg\].*\n')


def strip_script_line(line: str):
    """移除脚本行中不需要的部分。"""
    return STRIP_PATTERN.sub('#\n', line)


def strip_script(src_path: str, dst_path: str):
    """复制并移除脚本中部分内容。"""
    print("ybh2: " + dst_path)
    with open(dst_path, 'w', encoding='utf-8') as out_file:
        with open(src_path, encoding='utf-8') as in_file:
            for in_line in in_file:
                out_file.write(strip_script_line(in_line))


def strip_and_chmod_script(src_path: str, delivery_path: str):
    """复制并修改脚本权限。"""
    dst_path = os.path.join(
        delivery_path, os.path.basename(src_path)
    )

    if os.path.exists(dst_path):
        os.remove(dst_path)

    strip_script(src_path, dst_path)
    os.chmod(dst_path, 0o500)


def write_config_inc_var(name: str, package_attr: Dict, file: FileIO):
    """向config.inc文件写入变量。"""
    if name in package_attr:
        value = str(package_attr[name]).lower()
        file.write(f"{name.upper()}={value}\n")


def generate_config_inc(delivery_path: str, package_attr: Dict):
    """生成config.inc文件。"""
    year = datetime.now().year
    config_inc = os.path.join(delivery_path, 'config.inc')
    header = [
        '#!/bin/sh\n',
        '#----------------------------------------------------------------------------\n',
        f'# Copyright Huawei Technologies Co., Ltd. 2023-{year}. All rights reserved.\n',
        '#----------------------------------------------------------------------------\n',
        '\n',
    ]
    if os.path.isfile(config_inc):
        os.chmod(config_inc, 0o700)

    with open(config_inc, 'w', encoding='utf-8') as file:
        file.writelines(header)
        write_config_inc_var('parallel', package_attr, file)
        write_config_inc_var('parallel_limit', package_attr, file)
        write_config_inc_var('use_move', package_attr, file)

    os.chmod(config_inc, 0o500)


def copy_and_generate_scripts(top_dir: str, delivery_path: str, package_attr: Dict):
    """复制脚本文件。"""
    parser_script = os.path.join(
        top_dir, 'package/ops_transformer/scripts/install_common_parser.sh'
    )
    strip_and_chmod_script(parser_script, delivery_path)

    common_func_script = os.path.join(
        top_dir, 'package/ops_transformer/scripts/common_func_v2.inc'
    )
    strip_and_chmod_script(common_func_script, delivery_path)

    common_func_v3_script = os.path.join(
        top_dir, 'package/ops_transformer/scripts/common_func_v3.inc'
    )
    print("ybh: " + delivery_path)
    strip_and_chmod_script(common_func_v3_script, delivery_path)

    generate_config_inc(delivery_path, package_attr)
