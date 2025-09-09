#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""实用工具。"""

import json
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
from functools import partial
from pathlib import Path
from re import Match, Pattern
from subprocess import CompletedProcess
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union

import yaml

from .funcbase import lazy_eval, with_statement


def all_files_in_directory(directory: str, followlinks: bool = False) -> Iterator[str]:
    """目录下的所有文件。"""
    for root, dirnames, filenames in os.walk(directory, followlinks=followlinks):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            yield os.path.join(root, name)


def read_file(filepath: Union[str, Path]) -> str:
    """读取文件。"""
    with open(str(filepath), encoding='utf-8') as file:
        return file.read()


read_text = read_file


def iter_text(filepath: Union[str, Path]) -> Iterator[str]:
    """迭代文本。"""
    with open(str(filepath), encoding='utf-8') as file:
        yield from file


def read_file_lines(filepath: Union[str, Path]) -> List[str]:
    """读取文件行。"""
    with open(str(filepath), encoding='utf-8') as file:
        return file.readlines()


def read_binary_file(filepath: Union[str, Path]) -> bytes:
    """读取二进制文件。"""
    with open(str(filepath), 'rb') as file:
        return file.read()


def write_file(data: str, filepath: Union[str, Path]):
    """写入文件。"""
    with open(str(filepath), 'w', encoding='utf-8') as file:
        file.write(data)


def write_file_lines(lines: List[str], filepath: Union[str, Path]):
    """写入文件行。"""
    with open(str(filepath), 'w', encoding='utf-8') as file:
        file.writelines(lines)


def write_binary_file(data: bytes, filepath: Union[str, Path]):
    """写入二进制文件。"""
    with open(str(filepath), 'wb') as file:
        file.write(data)


def load_yaml(filepath: Union[str, Path]) -> Tuple[bool, Optional[Dict]]:
    """加载yaml配置。"""
    filepath = Path(filepath)
    if not filepath.is_file():
        return False, None

    with filepath.open(encoding='utf-8') as file:
        data = yaml.safe_load(file)

    return True, data


def load_yaml_exp(filepath: Union[str, Path]) -> Dict:
    """加载yaml配置。异常版本。"""
    with open(filepath, encoding='utf-8') as file:
        return yaml.safe_load(file)


def save_yaml(data: Dict, filepath: Union[str, Path]) -> bool:
    """保存yaml配置。"""
    filepath = Path(filepath)
    with filepath.open("w", encoding='utf-8') as file:
        yaml.safe_dump(data, file, allow_unicode=True)

    return True


def load_json(filepath: Union[str, Path], **kwargs) -> Any:
    """加载json配置。"""
    return next(
        with_statement(
            partial(open, filepath, encoding='utf-8'),
            lazy_eval(partial(json.load, **kwargs))
        )
    )


def save_json(data: Any, filepath: Union[str, Path], **kwargs):
    """保存json配置。"""
    return next(
        with_statement(
            partial(open, filepath, 'w', encoding='utf-8'),
            lazy_eval(partial(json.dump, data, **kwargs))
        )
    )


def get_top_dirpath() -> Path:
    """获取top目录路径。"""
    path = Path(__file__).resolve().parents[5]
    return path



def get_build_dirpath() -> Path:
    """获取build目录路径。"""
    path = Path(__file__).resolve().parents[2]
    return path


def get_delivery_dirpath() -> Path:
    """获取delivery目录路径。"""
    path = get_build_dirpath() / 'delivery'
    return path


def get_build_rule_dirpath() -> Path:
    """获取build_rule目录路径。"""
    path = get_build_dirpath() / 'version_build' / 'build_rule'
    return path


def get_current_dirpath(cur_file: str) -> Path:
    """获取当前目录路径。"""
    return Path(cur_file).resolve().parent


def get_cpu_count() -> int:
    """获取cpu数。"""
    return multiprocessing.cpu_count()


def shell_exec(cmd: str):
    """执行shell命令。"""
    print(cmd, flush=True)
    subprocess.run(cmd, check=True, shell=True)


def print_cmd(cmd: List[str]):
    """打印命令行。"""
    print(shlex.join(cmd), flush=True)


def _print_cmd_with_quiet(cmd: List[str], quiet: Optional[bool] = None):
    """打印命令行。"""
    if quiet is False:
        print_cmd(cmd)


def run_cmd(cmd: List[str],
            check: bool = True,
            quiet: bool = None,
            close_fds: bool = False,
            **kwargs) -> CompletedProcess:
    """执行命令。
    close_fds默认为False：工程脚本经常调用make，make需要通过文件句柄与父进程通信。
    shell中拉起子进程默认不会关闭文件句柄，与shell的行为保持一致。
    """
    _print_cmd_with_quiet(cmd, quiet)
    return subprocess.run(cmd, check=check, close_fds=close_fds, **kwargs)


def run_cmd_output(cmd: List[str],
                   check: bool = True,
                   quiet: bool = True,
                   **kwargs) -> List[str]:
    """执行命令，获取结果。"""
    result = run_cmd(cmd, check=check, quiet=quiet, **kwargs)
    return result.stdout.decode().splitlines()


def touch(path: str):
    """创建空文件。"""
    Path(path).touch()


def strip_suffix(suffix: str, string: str) -> str:
    """如果字符串的后缀存在，则删除。"""
    if string.endswith(suffix):
        return string[:-len(suffix)]
    return string


def get_available_tool(tools: Iterator[str]) -> Optional[str]:
    """获取可用工具。"""
    for tool in tools:
        tool_path = shutil.which(tool)
        if tool_path:
            return tool_path
    return None


class DictReplace(NamedTuple):
    """使用字典替换字符串。"""
    data: Dict[str, str]
    regex: Pattern

    def _replace_match(self, matchobj: Match) -> str:
        """替换匹配"""
        matchstr = matchobj.group(0)
        return self.data.get(matchstr, matchstr)

    def __call__(self, instr: str) -> str:
        """替换字符串。"""
        if not self.data:
            return instr
        return self.regex.sub(self._replace_match, instr)


def create_dict_replace(data: Dict[str, str], word: bool = True) -> DictReplace:
    """创建字典替换。"""
    ptn = "|".join(re.escape(item) for item in data)
    if word:
        ptn = f'\\b({ptn})\\b'
    regex = re.compile(ptn)
    return DictReplace(data, regex)


class Replacer(NamedTuple):
    """替换器。"""
    replace: DictReplace
    recover: DictReplace


def create_replacer(pairs: Iterable[Tuple[str, str]]) -> Replacer:
    """创建替换器。"""
    return Replacer(
        create_dict_replace(dict(pairs), False),
        create_dict_replace(dict(map(reversed, pairs)), False)
    )
