#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
build_opp_kernel_static.py
"""
import concurrent.futures
import glob
import multiprocessing
import os
import re
import platform
import stat
import json
import argparse
import subprocess
import logging as log
from pathlib import Path
from typing import Dict, List


class Const:
    zx_name = "opp_kernel_static"
    x86 = "x86_64"
    arm = "aarch64"
    ascendc_key = "NNOPBASE_"


def shell_exec(cmd, shell=False):
    try:
        ps = subprocess.Popen(cmd, shell)
        ps.communicate(timeout=180)
    except BaseException as e:
        log.error(f"shell_exec error: {e}")
        os._exit(1)


def shell_checkout_key_func(symbol_file, key_str):
    """
    shell执行，带管道功能，参考命令如下：
    cmd = f"nm {lib_shared} | awk '{{print $3}}' | xargs c++filt | grep 'optiling::op_impl_register_'"
    :param symbol_file:
    :param key_str:
    :return:
    """
    process = subprocess.Popen(("cat", symbol_file), stdout=subprocess.PIPE)
    awk_out = subprocess.check_output(("awk", "{print $8}"), stdin=process.stdout)
    process.wait()
    cppfilt = subprocess.check_output(("c++filt",), input=awk_out)
    if key_str not in cppfilt.decode("utf-8"):
        return "".encode("utf-8")
    grep_out = subprocess.check_output(("grep", key_str), input=cppfilt)
    return grep_out.decode("utf-8")


def to_upper_camel_case(x) -> str:
    """转大驼峰法命名"""
    s = re.sub('_([a-zA-Z])', lambda m: (m.group(1).upper()), x.lower())
    return s[0].upper() + s[1:]


def generate_symbol(args):
    library_file = args.library_file
    symbol_file = args.symbol_file
    if not os.path.exists(library_file):
        raise FileExistsError(f"generate_symbol input library error, file <{library_file}> not exists")
    process = subprocess.Popen(("readelf", "-Ws", library_file), stdout=subprocess.PIPE)
    output = process.communicate(timeout=180)[0].decode("utf-8")
    with open(symbol_file, "w") as fd:
        fd.write(output)


def parser_generate_symbol(subparsers):
    generate_symbol_parser = subparsers.add_parser(name='GenerateSymbol', help='Generate Symbol file of input library')
    generate_symbol_parser.add_argument('-l', '--library_file', type=str, required=False, dest="library_file",
                                        default="", help="Input the library file")
    generate_symbol_parser.add_argument('-s', '--symbol_file', type=str, required=False, dest="symbol_file", default="",
                                        help="The symbol file for output")
    generate_symbol_parser.set_defaults(func=generate_symbol)


class CompileOpStaticLib:
    def __init__(self, ops_compile_files: Dict, out_path: str, dist_index: int, arch: str):
        """
        分布式编译器上执行：编译静态库
        编译完成后根据输入的编译文本再生成静态.a文件
        :param op_binary_path:
        :param dist_index:
        :param arch: x86_64/aarch64
        """
        self.ops_compile_files = ops_compile_files
        self.out_path = out_path
        self.part_index = dist_index
        self.cpu_arch = arch
        if self.cpu_arch not in [Const.x86, Const.arm]:
            raise Exception(f"CompileOpStaticLib Error, input arch<{arch}> error...")


    def compile_link_single(self, file_path, file_o):
        (dir_path, file_name) = os.path.split(file_path)
        if self.cpu_arch == Const.x86:
            shell_exec(["bash", "-c", f"cd {dir_path} && "
                                      f"objcopy --input-target binary --output-target elf64-x86-64 "
                                      f"--binary-architecture i386 "
                                      f"{file_name} {file_o}"], shell=False)
        elif self.cpu_arch == Const.arm and platform.machine() != Const.x86:
            shell_exec(["bash", "-c", f"cd {dir_path} && "
                                      f"objcopy --input-target binary "
                                      f"--output-target elf64-littleaarch64 --binary-architecture aarch64 "
                                      f"{file_name} {file_o}"], shell=False)
        elif self.cpu_arch == Const.arm:
            shell_exec(["bash", "-c", f"cd {dir_path} && "
                                      f"aarch64-linux-gnu-objcopy --input-target binary "
                                      f"--output-target elf64-littleaarch64 --binary-architecture aarch64 "
                                      f"{file_name} {file_o}"], shell=False)


    def compile_link_o(self, out_path, file_path, is_need_path = True):
        """
        将json与.o文件，链接为：_json.o与_o.o文件
        :param bin_part: 绝对路径
        :param out_path: 输出路径，在ascendxxx同级目录下增加static目录
        :param file_pre: 文件名前缀
        :return:
        """
        file_pre = os.path.basename(file_path).replace('.', '_').replace('-', '_')
        path_o_prefix = os.path.join(out_path, f"data_{file_pre}_{self.cpu_arch}.o")
        # 向json文件中写入"filePath"参数
        if is_need_path and file_path.endswith(".json"):
            with open(file_path, 'r', encoding='UTF-8') as json_fd:
                json_dict = json.load(json_fd)
                json_dict["filePath"] = file_path.split("/bin/")[-1].split("/kernel/")[-1]
                file_path = os.path.join(out_path, os.path.basename(file_path))
                with open(file_path, 'w', encoding='UTF-8') as new_json_fd:
                    new_json_fd.write(json.dumps(json_dict, indent=4))
        self.compile_link_single(file_path, path_o_prefix)
        return

    def compile_ops_part_o(self, out_path):
        """
        将json与.o文件，链接为：_json.o与_o.o文件
        :param bin_part: 绝对路径
        :param out_path: 输出路径，在ascendxxx同级目录下增加static目录
        :param file_pre: 文件名前缀
        :return:
        """
        path_data_o = os.path.join(out_path, f"data_*{self.cpu_arch}.o")
        path_data_o_list = glob.glob(path_data_o)
        if not path_data_o_list:
            return
        (dir_path, ops_name) = os.path.split(out_path)
        file_part_o = f"{ops_name}_{self.cpu_arch}_part{self.part_index}.o"  # eg: floor_mod_aarch64_1.a
        path_part_o = os.path.join(dir_path, file_part_o)
        if self.cpu_arch == Const.x86 or (self.cpu_arch == Const.arm and platform.machine() != Const.x86):
            shell_exec(["bash", "-c", f"cd {out_path} && "
                                      f"ld -r {path_data_o} -o {path_part_o}"], shell=False)
        if self.cpu_arch == Const.arm and platform.machine() == Const.x86:
            shell_exec(["bash", "-c", f"cd {out_path} && "
                                      f"aarch64-linux-gnu-ld -r {path_data_o} -o {path_part_o}"], shell=False)
        return

    def exec_compile(self):
        """
        编译算子静态库
        :return:
        """
        def get_parallel_num() -> int:
            """
            获取多线程最大并发数量
            """
            num = multiprocessing.cpu_count() * 2
            if num == 0:
                num = 16
            return num

        job_num = get_parallel_num()
        for op in self.ops_compile_files:
            compile_files = self.ops_compile_files[op]["compile_files"]
            json_files = self.ops_compile_files[op]["json_files"]
            runtimeKB_jsons = self.ops_compile_files[op]["runtimeKB_json"]
            op_out_path = os.path.join(self.out_path, op)
            if not os.path.exists(op_out_path):
                os.makedirs(op_out_path, exist_ok=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=job_num) as executor:
                for file in compile_files:
                    executor.submit(self.compile_link_o, op_out_path, file)
                for file in json_files:
                    executor.submit(self.compile_link_o, op_out_path, file, False)
                for file in runtimeKB_jsons:
                    executor.submit(self.compile_link_o, op_out_path, file, False)

            with concurrent.futures.ThreadPoolExecutor(max_workers=job_num) as executor:
                executor.submit(self.compile_ops_part_o, op_out_path)
        return 0


def compile_static_library(args):
    index_num = args.index_num
    cpu_aarch = args.cpu_aarch

    if cpu_aarch not in [Const.x86, Const.arm]:
        raise Exception(f"Input cpu_aarch<{cpu_aarch}> Error, Please input parase")

    #aic*.json路径适配
    ops_info = os.path.join(args.build_dir, f"custom/op_impl/ai_core/tbe/config/{args.soc_version}/aic-{args.soc_version}-ops-info*.json")
    ops_info = glob.glob(ops_info)
    if hasattr(args, 'binary_path') and args.binary_path:
        binary_path = args.binary_path  
    else:
        binary_path = os.path.join(args.build_dir, f"binary/{args.soc_version}/bin")
    if hasattr(args, 'tuning_basic_path') and args.tuning_basic_path:
        tuning_basic_path = args.tuning_basic_path  
    else:
        tuning_basic_path = os.path.join(args.build_dir, f"tbe/config/{args.soc_version}")
    ops_compile_files = GenOpResourceIni(args.soc_version, args.build_dir, binary_path, ops_info, tuning_basic_path).ops_compile_files

    csl = CompileOpStaticLib(ops_compile_files, os.path.join(args.build_dir, f"bin_tmp/{args.soc_version}"), index_num, cpu_aarch)
    ret = csl.exec_compile()
    return ret


def parser_compile_static_library(subparsers):
    """ 配置静态编译参数及执行信息 """
    compile_lib_parser = subparsers.add_parser(name='StaticCompile',
                                    help='Compile static libraries(.a) on distributed server')
    compile_lib_parser.add_argument('-s', '--soc_version', type=str, required=True, dest="soc_version",
                                    help="Operator Name, eg: ascend910b, ascend310p")
    compile_lib_parser.add_argument('-b', '--build_dir', type=str, required=True, dest="build_dir",
                                    help="Input build dir for this project")
    compile_lib_parser.add_argument('-n', '--index_num', type=int, required=True, dest="index_num",
                                    help="Please input distributed compilation idx")
    compile_lib_parser.add_argument('-a', '--cpu_aarch', type=str, required=True, dest="cpu_aarch",
                                    help="Please input cpu aarch, eg:x86_64,aarch64")
    compile_lib_parser.add_argument('-B', '--binary_path', type=str, required=False, dest="binary_path",
                                    help="Optional binary path")
    compile_lib_parser.add_argument('-t', '--tuning_basic_path', type=str, required=False, dest="tuning_basic_path",
                                    help="Optional tuning basic path")
    compile_lib_parser.set_defaults(func=compile_static_library)


class GenOpResourceIni:
    def __init__(self, soc_version: str, build_dir: str, binary_path, ops_info, tuning_basic_path):
        self.soc_version = soc_version
        self.build_dir = build_dir
        self.tuning_json_path = None
        self.ops_compile_files: Dict = {}  # 定义头文件中可变部分
        self.self_def_ops: List = []
        self.ops_tuning_tiling: List = []
        self.op_symbols: Dict = {}
        self.tuning_basic_path = tuning_basic_path
        self.binary_path = binary_path
        self.ops_tuning_json4extern: Dict[str: list] = {}  # 知识库json文件名，用于生成头文件
        self.op_resource_path = os.path.join(self.build_dir, f"autogen/{self.soc_version}/aclnnop_resource")
        if len(ops_info) == 0:
            ops_info_json = {}
        else:
            with open(ops_info[0], "r") as autogen_fd:
                ops_info_json = json.load(autogen_fd)
        self.binary_op_list = []

        for ops in ops_info_json:
            self.ops_compile_files[ops] = {}
            self.ops_compile_files[ops]["compile_files"] = []
            self.ops_compile_files[ops]["json_files"] = []
            self.ops_compile_files[ops]["runtimeKB_json"] = []
            if 'opFile' in ops_info_json[ops]:
                json_file = f"{ops_info_json[ops]['opFile']['value']}.json"
            else:
                o_lists = list(Path(self.binary_path).rglob(f"{self.soc_version}/**/*{ops}*.o"))
                if len(o_lists) == 0:
                    continue
                else:
                    json_file = f"{os.path.basename(os.path.dirname(o_lists[0]))}.json"
            # 算子.json 路径适配
            json_path = os.path.join(self.binary_path, f"{json_file}")
            if not os.path.exists(json_path):
                continue
            with open(json_path, "r") as op_json_fd:
                op_json_content = json.load(op_json_fd)
            # 算子.json内 kernel json路径适配
            bin_json_file = os.path.join(self.binary_path, op_json_content["binList"][0]["binInfo"]["jsonFilePath"].split("/", 1)[1])
            ops_path = os.path.dirname(bin_json_file)
            self.ops_compile_files[ops]["ops_path"] = ops_path
            self.ops_compile_files[ops]["json_files"].append(json_path)
            self.ops_compile_files[ops]["compile_files"] = sorted([os.path.join(ops_path, file) for file in os.listdir(ops_path)])
            for kb_json in list(Path(self.tuning_basic_path).rglob(f"*_AiCore_{ops}_runtime_kb.json")):
                self.ops_compile_files[ops]["runtimeKB_json"].append(kb_json)
        self.checkout_self_def_ops()
        opapi_symbol = os.path.join(self.build_dir, "opapi_transformer.txt")
        if not os.path.exists(opapi_symbol):
            return
        # 查找l0接口
        l0op_list = shell_checkout_key_func(opapi_symbol, "_kernelName_Be_Defined_Multi_Times__")
        l0op_list = l0op_list.splitlines()
        self.binary_op_list = [op.split("::")[-1].split("_kernelName_")[0] for op in l0op_list]
        self.binary_op_list.sort()


    @staticmethod
    def generate_compile_file2extern(compile_files: List, json_files: List) -> str:
        """
        生成extern引用代码
        :param compile_files:
        :param json_files:
        :return:
        """
        ext_code = ""
        for file in json_files:
            name = file.split("/")[-1].replace(".", "_")
            ext_code = ext_code + f"""// {file.split("/")[-1]}
extern const uint8_t _binary_{name}_start[];
extern const uint8_t _binary_{name}_end[];
"""
        for file in compile_files:
            if not file.endswith(".json") and not file.endswith(".o"):
                continue
            if file.endswith("_failed.json"):
                continue
            name = file.split("/")[-1].replace(".", "_")
            ext_code = ext_code + f"""// {file}
extern const uint8_t _binary_{name}_start[];
extern const uint8_t _binary_{name}_end[];
"""
        return ext_code

    @staticmethod
    def generate_compile_file2tuple(op, compile_files: List, json_files: List) -> str:
        tuple_code = ""
        for file in json_files:
            name = file.split("/")[-1].replace(".", "_")
            tuple_code = tuple_code + f"""{{_binary_{name}_start, _binary_{name}_end}},
"""
        for file in compile_files:
            if not file.endswith(".json") and not file.endswith(".o"):
                continue
            if file.endswith("_failed.json"):
                continue
            name = file.split("/")[-1].replace(".", "_")
            tuple_code = tuple_code + f"""{{_binary_{name}_start, _binary_{name}_end}},
"""
        if tuple_code == "":
            return f"""
__attribute__((weak)) const OP_BINARY_RES& {op}KernelResource() {{
    static const OP_BINARY_RES resource = {{}};
    return resource;
}}
"""
        ret_code = f"""
__attribute__((weak)) const OP_BINARY_RES& {op}KernelResource() {{
    static const OP_BINARY_RES resource = {{{tuple_code[:-2]}}};
    return resource;
}}
"""
        return ret_code


    def gen_ops_ini_files(self):
        """
        解析autogen_op_files.json，生成算子头文件
        :return:
        """
        if not os.path.exists(self.op_resource_path):
            os.makedirs(self.op_resource_path)
        for ops in self.binary_op_list:
            if ops not in self.op_symbols:
                self.op_symbols[ops] = {}
                self.op_symbols[ops]["InferShape"] = []
                self.op_symbols[ops]["Tiling"] = []
                self.op_symbols[ops]["tuning"] = []
            self.op_symbols[ops]["InferShape"] = list(set(self.op_symbols[ops]["InferShape"]))
            self.op_symbols[ops]["Tiling"] = list(set(self.op_symbols[ops]["Tiling"]))
            self.op_symbols[ops]["tuning"] = list(set(self.op_symbols[ops]["tuning"]))
            compile_files = []
            json_files = []
            runtimeKB_jsons = []
            if ops in self.ops_compile_files:
                compile_files = self.ops_compile_files[ops]["compile_files"]
                json_files = self.ops_compile_files[ops]["json_files"]
                runtimeKB_jsons = self.ops_compile_files[ops]["runtimeKB_json"]
            ini_content = self.generate_op_resouce_ini(ops, compile_files, json_files, runtimeKB_jsons)

            ini_file = os.path.join(self.op_resource_path, f"{ops}_op_resource.cpp")
            if os.path.exists(ini_file):
                os.remove(ini_file)
            flags = os.O_WRONLY | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(ini_file, flags, modes), "w") as fd:
                fd.write(ini_content)

    def set_symbol(self, op_type, symbol_type, symbol):
        if not op_type in self.op_symbols:
            self.op_symbols[op_type] = {}
            self.op_symbols[op_type]["InferShape"] = []
            self.op_symbols[op_type]["Tiling"] = []
            self.op_symbols[op_type]["tuning"] = []
        self.op_symbols[op_type][symbol_type].append(symbol)


    def checkout_self_def_ops(self):
        # ophost txt 适配
        ophost_symbol = os.path.join(self.build_dir, "ophost_transformer.txt")
        if not os.path.exists(ophost_symbol):
            return
        # 查找自定义算子
        symbol_list = shell_checkout_key_func(ophost_symbol, "op_impl_register_")
        symbol_list = symbol_list.splitlines()
        for symbol in symbol_list:
            if "op_impl_register_infershape_" in symbol:
                op_type = symbol.split("op_impl_register_infershape_")[-1]
                self.set_symbol(op_type, "InferShape", symbol)
            elif "op_impl_register_optiling_" in symbol:
                op_type = symbol.split("op_impl_register_optiling_")[-1]
                self.set_symbol(op_type, "Tiling", symbol)
            else:
                op_type = symbol.split("op_impl_register_")[-1]
                index = -1
                for char in op_type[::-1]:
                    if not char.isdigit():
                        break
                    index = index - 1
                if index != -1:
                    index = index if op_type[index] != 'V' else index + 1
                    index = len(op_type) + index + 1
                    op_type = op_type[:index]
                if "optiling" in symbol:
                    self.set_symbol(op_type, "Tiling", symbol)
                elif "ops" in symbol:
                    self.set_symbol(op_type, "InferShape", symbol)
                else:
                    log.warning(f"Can't judge symbol type for {symbol}.")

        # 查找知识库函数
        tuning_ret = shell_checkout_key_func(ophost_symbol, "BankKeyRegistryInterf")
        tuning_tiling_list = tuning_ret.splitlines()
        for symbol in tuning_tiling_list:
            op_type = symbol.split("::")[-1].replace("g_", "").replace("BankKeyRegistryInterf", "")
            self.set_symbol(op_type, "tuning", symbol)
        tuning_ret = shell_checkout_key_func(ophost_symbol, "BankParseInterf")
        tuning_tiling_list = tuning_ret.splitlines()
        for symbol in tuning_tiling_list:
            op_type = symbol.split("::")[-1].replace("g_", "").replace("BankParseInterf", "")
            self.set_symbol(op_type, "tuning", symbol)
        tuning_ret = shell_checkout_key_func(ophost_symbol, "g_tuning_tiling_")
        tuning_tiling_list = tuning_ret.splitlines()
        for symbol in tuning_tiling_list:
            op_type = symbol.split("::")[-1].replace("g_tuning_tiling_", "").replace("Helper", "")
            self.set_symbol(op_type, "tuning", symbol)


    def generate_register_file2extern(self, op: str, type: str) -> str:
        if op in self.op_symbols:
            register_list = self.op_symbols[op][type]
        else:
            register_list = []
        ext_register = ""
        make_tuple_ops = ""
        for register_symbol in register_list:
            symbols = register_symbol.split("::")
            namespace = "::".join(symbols[:-1])
            ext_register = ext_register + f"""
namespace {namespace} {{
    extern gert::OpImplRegisterV2 {symbols[-1]};
}}
"""
            make_tuple_ops = make_tuple_ops + f"""
__attribute__((weak)) void * {op}{type}RegisterResource() {{
    return &{register_symbol};
}}
"""
        return (ext_register, make_tuple_ops)
    

    def generate_tuning_register_file2extern(self, op: str) -> str:
        if op in self.op_symbols:
            register_list = self.op_symbols[op]["tuning"]
        else:
            register_list = []
        ext_register = ""
        make_tuple_ops = ""
        tuning_symbols = {}
        for register_symbol in register_list:
            symbols = register_symbol.split("::")
            namespace = "::".join(symbols[:-1])
            type = f"{op}ClassHelper" if "g_tuning_tiling_" in register_symbol else "OpBankKeyFuncRegistryV2"
            ext_register = ext_register + f"""
namespace {namespace} {{
    class {type};
    extern {type} {symbols[-1]};
}}
"""
            if "g_tuning_tiling_" in register_symbol:
                tuning_symbols["helper"] = register_symbol
            elif "BankKeyRegistryInterf" in register_symbol:
                tuning_symbols["BankKeyRegistryInterf"] = register_symbol
            else :
                tuning_symbols["BankParseInterf"] = register_symbol
        
        for key in ["BankKeyRegistryInterf", "BankParseInterf", "helper"]:
            if key in tuning_symbols:
                make_tuple_ops = make_tuple_ops + f"&{tuning_symbols[key]}, "
            else:
                make_tuple_ops = make_tuple_ops + f"nullptr, "
        
        make_tuple_ops = f"""
__attribute__((weak)) void * {op}TuningRegisterResource() {{
    static std::vector<void *> resource = {{{make_tuple_ops[:-2]}}};
    return &resource;
}}
"""
        if len(register_list) == 0:
            ext_register = ""
            make_tuple_ops = f"""
__attribute__((weak)) void * {op}TuningRegisterResource() {{
    static std::vector<void *> resource = {{nullptr, nullptr, nullptr}};
    return &resource;
}}
"""
        return (ext_register, make_tuple_ops)

    def generate_tuningtiling_file2extern(self, op: str, runtimeKB_jsons: List) -> str:
        tunning_ext_code = ""
        if len(runtimeKB_jsons) == 0:
            return tunning_ext_code

        json_files_list = [os.path.basename(file) for file in runtimeKB_jsons]
        json_files_list.sort()
        for json_file in json_files_list:
            tunning_ext_code = tunning_ext_code + f"""// {json_file}
extern const uint8_t _binary_{json_file.replace('.', '_').replace('-', '_')}_start[];
extern const uint8_t _binary_{json_file.replace('.', '_').replace('-', '_')}_end[];
"""
        return f"""{tunning_ext_code}"""

    def generate_tuningtiling_file2tuple(self, op: str, runtimeKB_jsons: List) -> str:
        if len(runtimeKB_jsons) == 0:
            return f"""
__attribute__((weak)) const OP_RUNTIME_KB_RES& {op}TuningResource() {{
    static const OP_RUNTIME_KB_RES resource = {{}};
    return resource;
}}
"""
        tuple_code = ""
        json_files_list = [os.path.basename(file) for file in runtimeKB_jsons]
        json_files_list.sort()
        for json_file in json_files_list:
            tmp_name = json_file.replace('.', '_').replace('-', '_')
            tuple_code = tuple_code + f"""{{_binary_{tmp_name}_start, _binary_{tmp_name}_end}},
"""
        ret_code = f"""
__attribute__((weak)) const OP_RUNTIME_KB_RES& {op}TuningResource() {{
    static const OP_RUNTIME_KB_RES resource = {{{tuple_code[:-2]}}};
    return resource;
}}
"""
        return ret_code

    def generate_op_resouce_ini(self, op: str, compile_files: List, json_files: List, runtimeKB_jsons: List) -> str:
        """
        生成算子ini数据
        :param op:
        :param compile_files:
        :param json_files:
        :return:
        """
        (tiling_symbol, tiling_fuc) = self.generate_register_file2extern(op, "Tiling")
        (infer_symbol, infer_func) = self.generate_register_file2extern(op, "InferShape")
        (tuning_register_symbol, tuning_register_func) = self.generate_tuning_register_file2extern(op)
        tuning_extern_code = self.generate_tuningtiling_file2extern(op, runtimeKB_jsons)
        tuning_tuple_code = self.generate_tuningtiling_file2tuple(op, runtimeKB_jsons)
        ext_compile = self.generate_compile_file2extern(compile_files, json_files)
        binary_tuple_code = self.generate_compile_file2tuple(op, compile_files, json_files)

        if infer_symbol == "" and infer_func == "":
            infer_func = f"""
__attribute__((weak)) void * {op}InferShapeRegisterResource(){{
    return nullptr;
}}
"""
        if tiling_symbol == "" and tiling_fuc == "":
            tiling_symbol = f"""
namespace optiling {{
    extern gert::OpImplRegisterV2 op_impl_register_optiling_DefaultImpl;
}}
"""
            tiling_fuc = f"""
__attribute__((weak)) void * {op}TilingRegisterResource() {{
    return &optiling::op_impl_register_optiling_DefaultImpl;
}}
"""

        ini_content = f"""/******************{op}算子的所有资源**********************/
#include "register/op_impl_registry.h"
#include <vector>
#include <tuple>
#include <map>
#include <graph/ascend_string.h>

using OP_HOST_FUNC_HANDLE = std::vector<void *>;
using OP_RES = std::tuple<const uint8_t *, const uint8_t *>;
using OP_BINARY_RES = std::vector<OP_RES>;
using OP_RUNTIME_KB_RES = std::vector<OP_RES>;
using OP_RESOURCES  = std::map<ge::AscendString,
    std::tuple<OP_HOST_FUNC_HANDLE, OP_BINARY_RES, OP_RUNTIME_KB_RES>>;
{tiling_symbol}
{infer_symbol}
{tuning_extern_code}
{tuning_register_symbol}
// 二进制
{ext_compile}

namespace l0op {{
{tuning_register_func}
{tiling_fuc}
{infer_func}
{tuning_tuple_code}
{binary_tuple_code}
}}

"""
        return ini_content


def generate_op_resource_h_file(args):
    soc_version: str = args.soc_version
    build_dir = args.build_dir

    # aic*.json 适配
    ops_info = os.path.join(build_dir, f"custom/op_impl/ai_core/tbe/config/{soc_version}/aic-{soc_version}-ops-info*.json")
    ops_info = glob.glob(ops_info)
    if hasattr(args, 'binary_path') and args.binary_path:
        binary_path = args.binary_path  
    else:
        binary_path = os.path.join(args.build_dir, f"binary/{args.soc_version}/bin")
    if hasattr(args, 'tuning_basic_path') and args.tuning_basic_path:
        tuning_basic_path = args.tuning_basic_path  
    else:
        tuning_basic_path = os.path.join(args.build_dir, f"tbe/config/{args.soc_version}")

    gen_ini = GenOpResourceIni(soc_version, build_dir, binary_path, ops_info, tuning_basic_path)
    gen_ini.gen_ops_ini_files()
    return


def parser_generate_op_resource_h_file(subparsers):
    gen_resource_ini_parser = subparsers.add_parser(name='GenStaticOpResourceIni',
                                                    help='Generate xxx_op_resource.h on consolidation server')
    gen_resource_ini_parser.add_argument('-s', '--soc_version', type=str, required=True, dest="soc_version",
                                         help="Operator Name, eg: ascend910b, ascend310p")
    gen_resource_ini_parser.add_argument('-b', '--build_dir', type=str, required=True, dest="build_dir",
                                         help="Input build dir for this project")
    gen_resource_ini_parser.add_argument('-B', '--binary_path', type=str, required=False, dest="binary_path",
                                        help="Optional binary path")
    gen_resource_ini_parser.add_argument('-t', '--tuning_basic_path', type=str, required=False, dest="tuning_basic_path",
                                        help="Optional tuning basic path")
    gen_resource_ini_parser.set_defaults(func=generate_op_resource_h_file)


def execute_argus_parse_func():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='Subparsers Commands')

    """ 配置静态编译参数及执行信息 """
    parser_compile_static_library(subparsers)

    """ 配置头文件生成功能参数及执行信息 """
    parser_generate_op_resource_h_file(subparsers)

    """ 生成指定库的symbol文件 """
    parser_generate_symbol(subparsers)

    """ 执行函数功能 """
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    execute_argus_parse_func()
    exit(0)
