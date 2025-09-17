#!/usr/bin/python3
# -*- coding:utf-8 -*-
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
import sys
import numpy as np
import glob
import os

class _PrintColorPrefix:
    BLUE = "\033[1;34m"
    YELLOW = "\033[0;33m"
    RESET = "\033[0;0m"

class ColorPrint:
    @staticmethod
    def print_yellow(text):
        print(">> %s%s%s" % (_PrintColorPrefix.YELLOW, text, _PrintColorPrefix.RESET))
    
    @staticmethod
    def print_blue(text):
        print(">> %s%s%s" % (_PrintColorPrefix.BLUE, text, _PrintColorPrefix.RESET))


curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(golden_file, output_file, d_type):
    d_type_dict = {
        "int64": np.int64,
        "int32": np.int32
    }
    dtype = d_type_dict.get(d_type)

    output_golden = np.fromfile(golden_file, dtype)
    output_actual = np.fromfile(output_file, dtype)

    diff_lst = [True if x == 0 else False for x in np.abs(np.subtract(output_actual, output_golden))]
    if all(diff_lst):
        ColorPrint.print_blue("PASS")
    else:
        ColorPrint.print_blue("FAIL")
    

def process(d_type):
    golden_file = os.path.join(curr_dir, "output_golden.bin")
    output_file = os.path.join(curr_dir, "output_actual.bin")
    result = compare_data(golden_file, output_file, d_type)

if __name__ == '__main__':
    process(sys.argv[1])