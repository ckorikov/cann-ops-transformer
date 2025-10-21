#!/usr/bin/env python3
# coding: utf-8
#Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#This file is a part of the CANN Open Software.
#Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
#Please refer to the License for details. You may not use this file except in compliance with the License.
#THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#See LICENSE in the root of the software repository for the full text of the License.
import os
import sys
import shutil
import logging

def main():
    if len(sys.argv) != 3:
        logging.error("Usage: python clean_op_stub.py <op_name_list> <path>")
        sys.exit(1)

    op_names_raw = sys.argv[1]
    base_path = sys.argv[2]

    op_names = [op.strip() for op in op_names_raw.split(",") if op.strip()]
    if not op_names:
        logging.info("No valid op names provided.")
        sys.exit(0)

    ascend910b_dir = os.path.join(base_path, "opp", "built-in", "op_impl", "ai_core", "tbe", "kernel", "ascend910b")
    if not os.path.exists(ascend910b_dir):
        logging.warning("Base directory not found: %s", ascend910b_dir)
        sys.exit(0)

    for op in op_names:
        op_dir = os.path.join(ascend910b_dir, op)
        stub_file = os.path.join(op_dir, "opapi_ut.stub")

        if os.path.isdir(op_dir) and os.path.isfile(stub_file):
            logging.info("Removing directory: %s", op_dir)
            shutil.rmtree(op_dir)
        else:
            logging.info("No stub found or directory missing for op: %s, skip", op)

    lib_paths = [
        os.path.join(base_path, "opp", "built-in", "op_impl", "ai_core", "tbe", "op_tiling", "lib", "linux", "x86_64"),
        os.path.join(base_path, "opp", "built-in", "op_impl", "ai_core", "tbe", "op_tiling", "lib", "linux", "aarch64"),
    ]

    for lib_dir in lib_paths:
        lib_file = os.path.join(lib_dir, "libopmaster_rt2.0.so")
        if os.path.exists(lib_file):
            try:
                # 判断文件是否为0字节（即我们创建的占位文件）
                if os.path.getsize(lib_file) == 0:
                    os.remove(lib_file)
                    logging.info("[Removed] Empty placeholder file deleted: %s", lib_file)
                else:
                    logging.info("[Skip] %s is non-empty, keeping it.", lib_file)
            except Exception as e:
                logging.info("[Warning] Failed to process %s: %s", lib_file, e)

    logging.info("Cleanup finished.")

if __name__ == "__main__":
    main()
