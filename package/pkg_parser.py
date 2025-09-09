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

import copy
import glob
import json
import hashlib
import inspect
import itertools
import importlib.util
import argparse
import re
import xml.etree.ElementTree as ET
import os
import sys
from argparse import Namespace
from functools import partial
from io import StringIO
from itertools import chain, filterfalse
from operator import attrgetter, itemgetter, methodcaller
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple, Union
)

import pkg_utils
from filelist import FileItem, FileList, fill_is_common_path
from pkg_feature import (
    PkgFeature, combine_feature_and_feature_list, config_feature_to_set, feature_compatible,
    make_pkg_feature, pkg_feature_to_set,
)
from pkg_utils import (
    COMM_LOG, CONFIG_SCRIPT_PATH, ContainAsteriskError, FAIL, BLOCK_CONFIG_PATH,
    BLOCK_DEPENDTREE_CONF, BlockConfigError, EnvNotSupported, ErrMsgs, IllegalVersionDir,
    MultiPkgModError, MultiPkgSoftlinkError, PackageConfigError, PackageError,
    ParseOsArchError, USELESS_FILES, UnknownCopyTypeError, each_file_line,
    flatten, load_yaml, merge_dict, starpipe, strip_lines, yield_if
)
from utils.funcbase import constant, dispatch, invoke, pipe, star_apply
from version_info import (
    VersionInfo, VersionXml, get_version_dir, is_multi_version, parse_version_conf
)

THIS_FILE_NAME = __file__


MACHINE_ENV = '$(MACHINE_ENV)'

TARGET_ENV = '$(TARGET_ENV)'


# 环境变量字典
EnvDict = Dict[str, str]

# 文件信息
FileInfo = Dict[str, str]

# 包属性
PackageAttr = Dict[str, Union[str, bool]]

# 生成信息
GenerateInfo = Dict[str, str]

# spec文件信息
SpecInfo = Dict[str, str]

class ParseOption(NamedTuple):
    """解析参数。"""
    os_arch: Optional[str]
    pkg_version: Optional[str]
    build_type: Optional[str]
    package_check: bool
    ext_name: str = ''

    def release_type_matched(self, release_type: str) -> bool:
        """
        元素配置的release_type和打包配置的build_type是否匹配。

        如果release_type未配置，或build_type未配置，则不约束release_type（匹配）。
        """
        if not release_type:
            return True
        if not self.build_type:
            return True
        if self.build_type in release_type:
            return True
        return False

    def is_gcov_pkg(self) -> bool:
        """判断是否是gcov包。"""
        return 'gcov' in self.ext_name


def parse_os_arch(os_arch: str) -> Tuple[str, str, str]:
    """解析系统和架构。"""
    match = re.match("^([a-z]+)(\\d+(\\.\\d+)*)?[\\.-]?(\\S*)", os_arch)
    if match:
        os_name = match.group(1)
        os_ver = match.group(2)
        if match.group(4):
            arch = match.group(4)
        else:
            # 如果os_arch中没有配置ARCH，ARCH默认值为aarch64
            arch = 'aarch64'

        return os_name, os_ver, arch

    raise ParseOsArchError()


def replace_env(env_dict: EnvDict, in_str: str):
    """替换环境变量为实际值。"""
    env_list = re.findall(".*?\\$\\((.*?)\\).*?", in_str)
    for env in env_list:
        if env == 'FILE':
            continue
        if env in env_dict:
            if env_dict[env] is not None:
                in_str = in_str.replace(f"$({env})", env_dict[env])
            else:
                in_str = in_str.replace(f"$({env})", '')
        else:
            raise EnvNotSupported(f"Error: {env} not supported.")
    return in_str


class ParseEnv(NamedTuple):
    """解析上下文环境。"""
    need_package_func: Callable[[Dict], bool]
    env_dict: EnvDict
    parse_option: ParseOption
    delivery_dir: str
    top_dir: str
    query_feature: Callable = None

class BlockElement(NamedTuple):
    """块配置。"""
    name: str
    block_conf_path: str
    dst_path: str
    chips: Set[str]
    features: Set[str]
    pkg_features: Set[str]
    attrs: Dict[str, str]


# BlockElement直接透传给LoadedBlockElement的参数列表
BLOCK_ELEMENT_PASS_THROUGH_ARGS = ['dst_path', 'chips', 'features', 'pkg_features', 'attrs']


class LoadedBlockElement(NamedTuple):
    """加载后的块配置。"""
    root_ele: ET.Element
    use_move: bool
    dst_path: str
    chips: Set[str]
    features: Set[str]
    pkg_features: Set[str]
    attrs: Dict[str, str]


class FileInfoParsedResult(NamedTuple):
    """file_info元素解析结果。"""
    file_info: FileInfo
    move_infos: List[FileInfo]
    dir_infos: List[Dict[str, str]]
    expand_infos: List[Dict[str, str]]


class PkgMod(NamedTuple):
    """包内文件权限。"""
    path: str
    mod: int


class PkgSoftlink(NamedTuple):
    """包内软链接。"""
    dst_path: str
    src_path: str


class BlockConfig(NamedTuple):
    """块配置。"""

    dir_install_list: List[Dict]
    move_files: List[FileInfo]
    expand_content_list: List[Dict]
    package_content_list: List[Dict]
    pkg_mods: List[PkgMod]
    pkg_softlinks: List[PkgSoftlink]
    generate_infos: List[GenerateInfo]
    spec_infos: List[SpecInfo]


class PackerConfig(NamedTuple):
    """打包相关配置。"""
    fill_is_common_path: Callable[[FileList], Iterator[FileItem]]


class XmlConfig(NamedTuple):
    """打包xml配置。"""
    xml_relpath: str
    default_config: Dict[str, str]
    package_attr: PackageAttr
    version_info: VersionInfo
    blocks: List[BlockConfig]
    version: str
    version_xml: Optional[VersionXml]
    packer_config: PackerConfig

    def _collect_list(self, list_name):
        result = []
        for block in self.blocks:
            result.extend(getattr(block, list_name))
        return result

    @property
    def dir_install_list(self):
        return self._collect_list('dir_install_list')


    @property
    def move_content_list(self):
        return self._collect_list('move_files')

    @property
    def expand_content_list(self):
        return self._collect_list('expand_content_list')

    @property
    def package_content_list(self):
        return self._collect_list('package_content_list')

    @property
    def pkg_mods(self) -> List[PkgMod]:
        return self._collect_list('pkg_mods')

    @property
    def pkg_softlinks(self) -> List[PkgSoftlink]:
        return self._collect_list('pkg_softlinks')

    @property
    def generate_infos(self) -> List[GenerateInfo]:
        return self._collect_list('generate_infos')

    @property
    def spec_infos(self) -> List[SpecInfo]:
        return self._collect_list('spec_infos')

    @property
    def block_name(self):
        return self.default_config.get('name', '')



# 默认包属性
DEFAULT_PACKAGE_ATTR = {
    'gen_version_info': True,
}


def parse_package_info(package_info_ele: Optional[ET.Element]) -> Dict:
    """解析package_info元素。"""

    def get_package_info_attrs(ele: ET.Element) -> Iterator[Tuple[str, Union[str, bool]]]:
        # expand_asterisk: 展开配置中星号
        # parallel: 并行复制文件
        # parallel_limit: 限制并发数
        # package_check: 检查filelist.csv中配置目录是否完整
        # check_features: 检查filelist.csv中所有feature是否符合package_check
        # gen_version_info: 是否生成version.info文件
        bool_attrs = (
            'expand_asterisk', 'parallel', 'parallel_limit', 'package_check', 'check_features',
            'use_move', 'gen_version_info'
        )
        bool_values = ('t', 'true', 'y', 'yes')
        if ele.tag in bool_attrs:
            if ele.text.lower() in bool_values:
                yield ele.tag, True
            else:
                yield ele.tag, False
        else:
            if ele.tag == 'spc_version':
                yield ele.tag, ele.text
                if ele.attrib.get('show_in_pkg_name', '').lower() in bool_values:
                    yield 'spc_version_show_in_pkg_name', True
            else:
                yield ele.tag, ele.text

    if not package_info_ele:
        return {}

    attr = dict(
        chain.from_iterable(map(get_package_info_attrs, list(package_info_ele)))
    )

    return attr


def parse_package_attr_by_args(args: Namespace) -> Dict:
    """通过命令行参数解析"""
    def pairs():
        if hasattr(args, 'chip_name') and args.chip_name:
            yield 'chip_name', args.chip_name
        if hasattr(args, 'suffix') and args.suffix:
            yield 'suffix', args.suffix
        if hasattr(args, 'func_name') and args.func_name:
            yield 'func_name', args.func_name
    return dict(pairs())


def parse_package_attr(root_ele: ET.Element, args: Namespace) -> Dict:
    """通过根元素解析package_info元素。"""
    package_info_ele = root_ele.find("package_info")
    return merge_dict(
        DEFAULT_PACKAGE_ATTR,
        parse_package_info(package_info_ele),
        parse_package_attr_by_args(args),
    )


def parse_feature_config_by_root(root_ele: ET.Element) -> Dict:
    """通过根元素解析feature_config元素。"""
    package_info_ele = root_ele.find("feature_config")
    return parse_package_info(package_info_ele)


def is_version_xml(filepath: str) -> bool:
    """是否为xml版本配置。"""
    if os.path.isfile(filepath) and VersionXml.match(filepath):
        return True
    return False


def parse_version_xml(filepath: str, top_dir: str) -> Tuple[bool, Optional[VersionXml]]:
    """解析版本配置。"""
    full_path = os.path.join(top_dir, filepath)
    if is_version_xml(full_path):
        version_xml = VersionXml.parse(full_path, top_dir)
        return True, version_xml

    return False, None


def parse_version(version: str, top_dir: str) -> Tuple[str, Optional[VersionXml]]:
    """解析版本号。"""
    if not version:
        return version, None
    ret, version_xml = parse_version_xml(version, top_dir)
    if ret:
        return version_xml.get_release_version(), version_xml
    return parse_version_conf(top_dir, version), None


def parse_version_by_package_info_and_option(parse_option: ParseOption,
                                             package_attr: PackageAttr,
                                             top_dir: str) -> Tuple[str, Optional[VersionXml]]:
    """解析版本号。"""
    if parse_option.pkg_version:
        return parse_version(parse_option.pkg_version, top_dir)

    return parse_version(package_attr.get('version'), top_dir)


def get_feature_list(args: argparse.Namespace, feature_attr: Dict) -> str:
    """获取feature.list配置"""
    return args.feature_list or feature_attr.get('feature_list', '')


def remove_suffix_x(version_prefix: str) -> str:
    """移除版本号后缀的x。"""
    if version_prefix.endswith('x'):
        return version_prefix[:-1]
    return version_prefix


def get_prefix_length(version: str, version_prefix: str) -> int:
    """获取版本前缀长度。"""
    if version.startswith(version_prefix):
        return len(version_prefix)
    return 0


def get_compat_ver(version: str, versions_path: str) -> str:
    """获取兼容版本。

    等价的过程代码为:
    data = load_yaml(versions_path)
    max_len = 0
    max_value = None
    for name, value in data.items():
        name = remove_suffix_x(name)
        prefix_length = get_prefix_length(version, name)
        if prefix_length > max_len:
            max_len = prefix_length
            max_value = value
    return max_value['compatible_with']

    等价的pipe代码为:
    """
    get_func = pipe(
        load_yaml,
        methodcaller('items'),
        partial(
            map,
            dispatch(
                pipe(itemgetter(0), remove_suffix_x, partial(get_prefix_length, version)),
                itemgetter(1),
            ),
        ),
        partial(map, tuple),
        partial(max, key=itemgetter(0)),
        itemgetter(1),
        itemgetter('compatible_with'),
    )
    return get_func(versions_path)


def get_compat_ver_by_top_dir(version: str, top_dir: str) -> Optional[str]:
    """获取兼容版本号。"""
    versions_path = os.path.join(
        top_dir, 'build', 'release', 'config', 'common', 'compatible_versions.yml'
    )
    if os.path.exists(versions_path):
        return get_compat_ver(version, versions_path)
    return None


def render_cann_version(a_ver: int,
                        b_ver: int,
                        c_ver: Optional[int],
                        d_ver: Optional[int],
                        e_ver: Optional[int],
                        f_ver: Optional[int]) -> str:
    """渲染CANN版本号。"""
    buffer = StringIO()
    buffer.write('(')
    buffer.write(f'({a_ver + 1} * 100000000) + ({b_ver + 1} * 1000000)')
    if c_ver is not None:
        buffer.write(f' + ({c_ver + 1} * 10000)')
    if d_ver is not None:
        buffer.write(f' + (({d_ver + 1} * 100) + 5000)')
    if e_ver is not None:
        buffer.write(f' + ({e_ver + 1} * 100)')
    if f_ver is not None:
        buffer.write(f' + {f_ver}')
    buffer.write(')')
    return buffer.getvalue()


def get_cann_version(version_dir: str) -> str:
    """获取CANN版本号。"""
    if not version_dir:
        return 0

    release_pattern = re.compile(r'(\d+)\.(\d+)\.(\d+)', re.IGNORECASE)
    matched = release_pattern.fullmatch(version_dir)
    if matched:
        return render_cann_version(
            int(matched.group(1)), int(matched.group(2)), int(matched.group(3)), None, None, None
        )

    release_alpha_pattern = re.compile(r'(\d+)\.(\d+)\.(\d+)\.[a-z]*(\d+)', re.IGNORECASE)
    matched = release_alpha_pattern.fullmatch(version_dir)
    if matched:
        return render_cann_version(
            int(matched.group(1)), int(matched.group(2)), int(matched.group(3)), None, None,
            int(matched.group(4))
        )

    rc_pattern = re.compile(r'(\d+)\.(\d+)\.RC(\d+)', re.IGNORECASE)
    matched = rc_pattern.fullmatch(version_dir)
    if matched:
        return render_cann_version(
            int(matched.group(1)), int(matched.group(2)), None, int(matched.group(3)), None, None
        )

    test_pattern = re.compile(r'(\d+)\.(\d+)\.T(\d+)', re.IGNORECASE)
    matched = test_pattern.fullmatch(version_dir)
    if matched:
        return render_cann_version(
            int(matched.group(1)), int(matched.group(2)), None, None, int(matched.group(3)), None
        )

    alpha_pattern = re.compile(r'(\d+)\.(\d+)\.RC(\d+)\.[a-z]*(\d+)', re.IGNORECASE)
    matched = alpha_pattern.fullmatch(version_dir)
    if matched:
        return render_cann_version(
            int(matched.group(1)), int(matched.group(2)), None, int(matched.group(3)), None,
            int(matched.group(4))
        )

    cann_pattern = re.compile(r'CANN-(\d+)\.(\d+)', re.IGNORECASE)
    matched = cann_pattern.fullmatch(version_dir)
    if matched:
        return render_cann_version(
            int(matched.group(1)), int(matched.group(2)), None, None, None, None
        )

    raise IllegalVersionDir(version_dir)


def get_cann_version_info(name: str, version_dir: str) -> List[Tuple[str, str]]:
    """获取CANN版本号信息。"""
    version_info = []

    # 删除字符串中的_VERSION
    package_name = name[:-8]

    if not version_dir:
        version_str = '0'
    elif version_dir.startswith('CANN-'):
        version_str = version_dir[5:]
    else:
        version_str = version_dir

    version_info.append((f'{package_name}_VERSION_STR', f'"{version_str}"'))

    version = get_cann_version(version_dir)
    version_info.append((f'{package_name}_VERSION', version))

    return version_info


def get_default_env_items() -> Iterator[Tuple[str, str]]:
    """获取默认环境字典条目。"""
    yield ('VERSION_DIR', '')
    yield ('HOME', os.environ.get('HOME'))


def get_env_itmes_by_version(version: Optional[str]) -> Iterator[Tuple[str, str]]:
    """根据version获取环境字典条目。"""
    if version:
        yield ('ASCEND_VER', version)

        version_parts = version.split('.')
        for idx in range(1, len(version_parts)+1):
            yield (f'CUR_VER[{idx}]', '.'.join(version_parts[:idx]))
        yield ('CUR_VER', version)
        yield ('LOWER_CUR_VER', version.lower())


def get_env_itmes_by_version_dir(version_dir: Optional[str]) -> Iterator[Tuple[str, str]]:
    """根据version_dir获取环境字典条目。"""
    if version_dir:
        yield ('VERSION_DIR', version_dir)


def get_os_arch_default_env_items() -> Iterator[Tuple[str, str]]:
    """获取系统相关默认环境字典条目。"""
    yield ('OS_NAME', 'linux')
    yield ('OS_VER', '')
    yield ('ARM', 'aarch64')
    yield ('TARGET_ENV', '$(TARGET_ENV)')
    yield ('MACHINE_ENV', '$(MACHINE_ENV)')


def get_env_items_by_os_arch(os_arch: str) -> Iterator[Tuple[str, str]]:
    """根据os_arch获取环境字典条目。"""
    if os_arch:
        os_name, os_ver, arch = parse_os_arch(os_arch)
        yield ('OS_NAME', os_name)
        yield ('OS_VER', os_ver)
        yield ('ARCH', arch)
        yield ('OS_ARCH', os_arch)
        if arch in ('arm', 'sw_64'):
            yield ('ARM', arch)
        else:
            yield ('ARM', 'aarch64')
        yield ('TARGET_ENV', f"{arch}-linux")
        yield ('MACHINE_ENV', f"{arch}")
    else:
        yield from get_os_arch_default_env_items()


def get_env_items_by_timestamp(timestamp: Optional[str]) -> Iterator[Tuple[str, str]]:
    """根据timestamp获取环境字典条目。"""
    if timestamp:
        yield ('TIMESTAMP', timestamp)
        yield ('TIMESTAMP_NO', timestamp.replace('_', ''))
    else:
        yield ('TIMESTAMP', '0')
        yield ('TIMESTAMP_NO', '0')


def parse_env_dict(os_arch: str,
                   package_attr: PackageAttr,
                   version: Optional[str],
                   version_dir: Optional[str],
                   timestamp: Optional[str],
                   compat_ver: Optional[str]) -> EnvDict:
    """解析环境变量字典。"""
    env_dict = dict(
        chain(
            get_default_env_items(),
            yield_if(('ARCH', package_attr.get('default_arch')), itemgetter(1)),
            get_env_items_by_os_arch(os_arch),
            get_env_itmes_by_version(version),
            get_env_itmes_by_version_dir(version_dir),
            yield_if(('VERSION_DIR', version_dir), constant(version_dir)),
            get_env_items_by_timestamp(timestamp),
            yield_if(('COMPAT_VER', compat_ver), constant(compat_ver)),
        )
    )

    return env_dict


def parse_install_version_info(attr_info: Dict[str, str]) -> Tuple[bool, Dict]:
    """解析install_version_info。"""
    if attr_info.get('install') == 'true':
        return True, attr_info
    return False, None


# 跳过空行
skip_empty_lines = partial(filter, bool)

# 跳过注释行
skip_comment_lines = partial(filterfalse, methodcaller('startswith', '#'))

# 跳过空行和注释行
skip_empty_and_comment_lines = pipe(
    skip_empty_lines,
    skip_comment_lines,
)


def load_itf_version_conf(itf_conf: str,
                          replace_env_func: Callable[[str], str]) -> List[str]:
    """加载接口版本配置。"""
    load_func = pipe(
        each_file_line,
        strip_lines,
        skip_empty_and_comment_lines,
        partial(map, replace_env_func),
        list,
    )
    return load_func(itf_conf)


def parse_itf_version_paths(version_info_ele: Optional[ET.Element]
                            ) -> Tuple[ErrMsgs, List[str]]:
    """解析接口版本路径列表。"""
    def has_itf_version(ele: ET.Element) -> bool:
        return bool(ele.attrib.get('itf_version'))

    if version_info_ele is None:
        return [], []

    interface_eles = version_info_ele.findall('./interface')

    err_msgs = [
        'interface element doesn\'t have itf_version attribute!'
        for ele in interface_eles if not has_itf_version(ele)
    ]

    itf_version_paths = [
        ele.attrib['itf_version']
        for ele in interface_eles if has_itf_version(ele)
    ]

    return err_msgs, itf_version_paths



def parse_itf_versions(version_info_ele: Optional[ET.Element],
                       top_dir: str,
                       replace_env_func: Callable[[str], str]) -> Tuple[ErrMsgs, List[str]]:
    """解析itf_versions。"""

    def itf_conf_exists(path: str) -> bool:
        return os.path.isfile(os.path.join(top_dir, path))

    err_msgs, itf_version_paths = parse_itf_version_paths(version_info_ele)
    if err_msgs:
        return err_msgs, []

    err_msgs = [
        f'{path} doesn\'t exist!'
        for path in itf_version_paths if not itf_conf_exists(path)
    ]
    if err_msgs:
        return err_msgs, []

    itf_versions = flatten([
        load_itf_version_conf(os.path.join(top_dir, path), replace_env_func)
        for path in itf_version_paths
    ])

    return err_msgs, itf_versions


def get_base_version(args: Namespace, package_attr: PackageAttr) -> str:
    """获取基础版本。"""
    return args.base_version or package_attr.get('base_version')


def get_spc_version(args: Namespace, package_attr: PackageAttr) -> str:
    """获取基础版本。"""
    return args.base_version or package_attr.get('spc_version')


def get_timestamp(args: Namespace) -> Optional[str]:
    """获取触发时间戳。"""
    if 'tag' not in args or 'buildinfo_path' not in args:
        return None

    tag = args.tag
    if tag:
        timestamp_re = r"\d{8}_\d{9}"
        timestamp_list = re.findall(timestamp_re, tag)
        if not timestamp_list:
            raise PackageError("The {} format is incorrect.".format(tag))
        timestamp = timestamp_list[-1]
    elif args.buildinfo_path:
        buildinfo_dict = load_yaml(args.buildinfo_path)
        if 'params' in buildinfo_dict:
            timestamp = buildinfo_dict['params'].get('timestamp')
    else:
        timestamp = None
    return  timestamp


def parse_version_info_by_root(root_ele: ET.Element,
                               env_dict: EnvDict,
                               version: str,
                               version_xml: Optional[VersionXml],
                               base_version: str,
                               spc_version: str,
                               timestamp: str,
                               ) -> Tuple[ErrMsgs, VersionInfo]:
    """解析version_info元素。"""
    version_info_ele = root_ele.find('version_info')
    evaluate_info_func = partial(
        evaluate_info, loaded_block=make_loaded_block_element(root_ele), env_dict=env_dict
    )

    if version_info_ele is None:
        err_msgs, itf_versions = [], []
        attr_info = {}
        dst_path = ''
    else:
        err_msgs, itf_versions = parse_itf_versions(
            version_info_ele, pkg_utils.TOP_SOURCE_DIR, partial(replace_env, env_dict)
        )
        attr_info = evaluate_info_func(version_info_ele.attrib)
        dst_path = attr_info.get('dst_path', '')

    version_info = VersionInfo(
        dst_path,
        *parse_install_version_info(attr_info),
        itf_versions,
        version,
        version_xml,
        base_version,
        spc_version,
        timestamp
    )

    return err_msgs, version_info


def extract_element_attrib(ele: ET.Element) -> Dict:
    """提取元素属性。"""
    return ele.attrib.copy()


def extract_generate_info_content(generate_info_ele: ET.Element, env_dict: EnvDict) -> Dict:
    """提取生成信息内容。"""
    file_content = {
        sub_item.tag: replace_env(env_dict, sub_item.text)
        for sub_item in list(generate_info_ele)
    }
    return {
        'content': file_content
    }


def parse_generate_infos_by_loaded_block(loaded_block: LoadedBlockElement,
                                         default_config: Dict[str, str],
                                         env_dict: EnvDict,
                                         need_package_func: Callable[[Dict], bool]) -> List[Dict]:
    """根据根元素解析生成信息列表。"""
    return invoke(
        pipe(
            partial(
                map, pipe(
                    dispatch(
                        pipe(
                            extract_element_attrib,
                            partial(merge_dict, default_config),
                            partial(evaluate_info, loaded_block=loaded_block, env_dict=env_dict),
                        ),
                        partial(extract_generate_info_content, env_dict=env_dict),
                    ),
                    star_apply(merge_dict),
                )
            ),
            partial(filter, need_package_func),
            list,
        ),
        loaded_block.root_ele.findall('generate_info')
    )


def parse_spec_infos(loaded_block: LoadedBlockElement) -> List[Dict]:
    """根据根元素解析生成信息列表。"""
    spec_info_ele = loaded_block.root_ele.find('spec_info')
    if spec_info_ele is None:
        return []
    spec_infos = []
    for file in spec_info_ele.findall("file"):
        if "install_type" not in file.attrib:
            file.attrib["install_type"] = "all"
        spec_infos.append(spec_info_ele.attrib | file.attrib)
    return spec_infos


def join_pkg_inner_softlink(link_str_list: List[str]) -> str:
    """合并pkg_inner_softlink"""
    path = "/".join(link_str_list)
    return os.path.normpath(path)


def check_contain_asterisk(value: str) -> bool:
    """检查串是否包含星号。"""
    if '*' in value:
        return True
    return False


def check_value(value: str,
                package_check: bool,
                package_attr: PackageAttr):
    """检查元素value属性。"""
    if package_check and package_attr.get('suffix') == 'run':
        if check_contain_asterisk(value):
            raise ContainAsteriskError(value)


def arch_matched(info: Dict[str, str], env_dict: EnvDict) -> bool:
    """架构是否配套。"""
    cur_arch = make_pkg_feature(pkg_feature_to_set(env_dict.get('ARCH')), False)
    tgt_arch = make_pkg_feature(pkg_feature_to_set(info.get('arch', None)), False)
    return feature_compatible(cur_arch, tgt_arch)


def need_package(info: Dict[str, str],
                 input_feature: PkgFeature,
                 parse_option: ParseOption,
                 env_dict: EnvDict) -> bool:
    """是否需要打包。"""
    tgt_os_type = info.get('os_type', None)
    cur_os_type = env_dict.get('OS_NAME')
    if tgt_os_type and cur_os_type and tgt_os_type != cur_os_type:
        return False

    if not arch_matched(info, env_dict):
        return False

    pkg_features = info['pkg_feature']
    if not input_feature.matched(pkg_features):
        return False

    release_type = info.get('release_type', '').lower()
    if not parse_option.release_type_matched(release_type):
        return False

    if parse_option.is_gcov_pkg():
        gcov = info.get('gcov', '').strip().lower()
        if gcov == "false":
            return False

    return True


def get_src_prefix(file_info: FileInfo, env: ParseEnv) -> str:
    """获取文件的前缀。"""
    copy_type = file_info.get('copy_type')
    src_path = file_info['src_path']
    if copy_type == 'delivery':
        return os.path.join(env.delivery_dir, src_path)

    if copy_type == 'source':
        return os.path.join(env.top_dir, src_path)

    raise UnknownCopyTypeError(f'unknown copy_type {copy_type}')


def get_src_target(file_info: FileInfo, env: ParseEnv) -> str:
    """获取文件的实际路径。"""
    src_prefix = get_src_prefix(file_info, env)
    return os.path.join(src_prefix, file_info.get('value'))


def make_hash(filepath: str) -> str:
    """计算文件的hash(sha256)值。"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as file:
        sha256_hash.update(file.read())

    return sha256_hash.hexdigest()


def config_hash(parsed_result: FileInfoParsedResult, env: ParseEnv):
    """配置hash值。"""
    file_info = parsed_result.file_info
    # 如果配置了configurable，需要计算文件的hash值
    if file_info and file_info['configurable'] == 'TRUE':
        src_target = get_src_target(file_info, env)
        hash_value = make_hash(src_target)
        file_info['hash'] = hash_value
    return parsed_result


def apply_func(func: Callable[[str], str],
               value: Union[List[str], Set[str], str]
               ) -> Union[List[str], Set[str], str]:
    """对一个字符串，或字符串序列，应用函数。"""
    # 如：pkg_softlink列表
    if isinstance(value, list):
        return list(map(func, value))
    # 如：feature集合
    if isinstance(value, set):
        return set(map(func, value))
    return func(value)


REAL_PREFIX = 'real:'

def join_dst_path(base: str, other: str) -> str:
    """联结dst_path。"""
    if other.startswith('real:'):
        other = other[len(REAL_PREFIX):]
        return other
    return os.path.join(base, other)


def evaluate_info(info: Dict[str, str],
                  loaded_block: LoadedBlockElement,
                  env_dict: EnvDict,
                  pkg: bool = False) -> Dict[str, str]:
    """info元素求值。

    pkg标记是否为处理打包元素，例如pkg_mod，pkg_softlink。
    """

    if pkg:
        dst_keys = ('src_path', 'value')
    else:
        dst_keys = ('dst_path', 'pkg_softlink')

    replace_env_func = partial(replace_env, env_dict)
    add_dst_path_func = partial(join_dst_path, loaded_block.dst_path)

    def split_pkg_softlink(key: str, value: str) -> Tuple[str, str]:
        if key == 'pkg_softlink':
            return key, value.split(';')
        return key, value

    def upper_value(key: str, value: str) -> Tuple[str, str]:
        if key == 'configurable':
            return key, value.upper()
        return key, value

    def add_dst_path(key: str, value: str) -> Tuple[str, str]:
        if key in dst_keys:
            return key, apply_func(add_dst_path_func, value)
        return key, value

    def replace_pkg_inner_softlink(key: str, value: str) -> Tuple[str, str]:
        if key == 'pkg_inner_softlink':
            inner_softlink_new = [
                join_pkg_inner_softlink(link_str.split(':'))
                for link_str in value.split(';')
                if link_str.split(':')[0] == loaded_block.dst_path
            ]
            if inner_softlink_new:
                return key, ';'.join(inner_softlink_new)
            return key, 'NA'
        return key, value

    def merge_feature(key: str, value: str) -> Tuple[str, Set[str]]:
        if key in ('chip', 'feature', 'pkg_feature'):
            config_features = config_feature_to_set(value, key)
            return key, config_features | getattr(loaded_block, f'{key}s')
        return key, value

    def eval_value(_key: str, value: str) -> str:
        if value is None:
            return None
        return apply_func(replace_env_func, value)

    eval_value_func = starpipe(
        upper_value, split_pkg_softlink, add_dst_path, replace_pkg_inner_softlink,
        merge_feature, eval_value,
    )

    return {
        key: eval_value_func(key, value)
        for key, value in
        itertools.chain(
            # 默认值配置
            [
                ('dst_path', ''), ('configurable', 'FALSE'),
                ('chip', None), ('feature', None), ('pkg_feature', None),
            ],
            info.items()
        )
    }


def parse_dir_info_elements(loaded_block: LoadedBlockElement,
                            default_config: Dict[str, str],
                            package_attr: PackageAttr,
                            env: ParseEnv) -> List[Dict[str, str]]:
    """解析dir_info元素。"""
    dir_info_eles: List[ET.Element] = loaded_block.root_ele.findall('dir_info')
    dir_infos = []
    for item in dir_info_eles:
        dir_config = default_config.copy()
        dir_config.update(item.attrib)
        dir_config['module'] = dir_config.get('value')
        for sub_item in list(item):
            dir_info = dir_config.copy()
            dir_info.update(sub_item.attrib)
            dir_info = evaluate_info(dir_info, loaded_block, env.env_dict)
            check_value(
                dir_info['value'], env.parse_option.package_check, package_attr
            )

            if not env.need_package_func(dir_info):
                continue

            dir_infos.append(dir_info)

    return dir_infos


def expand_dir(file_info: FileInfo, get_src_target_func: Callable[[FileInfo], str]):
    """
    如果file_info中配置的路径是文件夹，需要展开到文件
    """
    file_info_list = []
    dir_info_list = []
    src_target = get_src_target_func(file_info)

    value_list = file_info.get('value').split('/')
    target_name = value_list[-1] if value_list[-1] else value_list[-2]

    # 这里把当前目录也加入到dir_info_list中
    dir_info_copy = file_info.copy()
    dir_info_copy['module'] = file_info.get('value')
    dir_info_copy['value'] = os.path.join(
        file_info.get('install_path', ''), target_name
    )

    # 子目录的权限按照xml中subdir_mod配置，如果没有配置subdir_mod按照install_mod配置
    subdir_mod = file_info.get("subdir_mod", None)
    if subdir_mod is not None:
        dir_info_copy['install_mod'] = subdir_mod
    # 被展开的当前目录不需要设置softlink
    dir_info_copy['install_softlink'] = 'NA'
    dir_info_copy['pkg_inner_softlink'] = 'NA'
    dir_info_list.append(dir_info_copy)

    for root, dirs, files in os.walk(src_target, followlinks=True):
        for useless_file in USELESS_FILES:
            if useless_file in dirs:
                dirs.remove(useless_file)
            if useless_file in files:
                files.remove(useless_file)
        # 不同操作系统上，os.walk遍历的结果顺序会略有不同，这里按字母排序，保证不同系统一致
        dirs.sort()
        files.sort()

        dirs_to_remove = []
        for name in dirs:
            dirname = os.path.join(root, name)
            if os.path.islink(dirname) and not need_dereference(file_info):
                # 如果是指向目录的软连接，则按照文件处理，无需在安装时创建目录，只需要卸载时删除就行
                relative_filename = os.path.relpath(dirname, src_target)
                relative_dir_name = os.path.split(relative_filename)[0]
                copy_file_info = file_info.copy()
                copy_file_info['value'] = name
                copy_file_info['src_path'] = os.path.join(
                    file_info['src_path'], file_info['value'], relative_dir_name
                )
                copy_file_info['dst_path'] = os.path.join(
                    file_info['dst_path'], target_name, relative_dir_name
                )
                copy_file_info['install_path'] = os.path.join(
                    file_info.get('install_path', ''), target_name, relative_dir_name
                )
                # 被展开的子文件不需要设置softlink
                copy_file_info['install_softlink'] = 'NA'
                copy_file_info['pkg_inner_softlink'] = 'NA'
                file_info_list.append(copy_file_info)
                dirs_to_remove.append(name)
                continue
            relative_dirname = os.path.relpath(dirname, src_target)
            dir_info_copy = file_info.copy()
            dir_info_copy['module'] = file_info.get('value')
            dir_info_copy['value'] = os.path.join(
                file_info.get('install_path', ''), target_name, relative_dirname
            )
            # 被展开的子目录不需要设置softlink
            dir_info_copy['install_softlink'] = 'NA'
            dir_info_copy['pkg_inner_softlink'] = 'NA'
            # 子目录的权限按照xml中subdir_mod配置，如果没有配置subdir_mod按照install_mod配置
            subdir_mod = file_info.get("subdir_mod", None)
            if subdir_mod is not None:
                dir_info_copy['install_mod'] = subdir_mod
            dir_info_list.append(dir_info_copy)
        for name in files:
            filename = os.path.join(root, name)
            relative_filename = os.path.relpath(filename, src_target)
            relative_dir_name = os.path.split(relative_filename)[0]
            copy_file_info = file_info.copy()
            # 将遍历到的子文件的相对目录加到对应的src_path,dst_path,install_path中
            copy_file_info['value'] = name
            copy_file_info['src_path'] = os.path.join(
                file_info['src_path'], file_info['value'], relative_dir_name
            )
            copy_file_info['dst_path'] = os.path.join(
                file_info['dst_path'], target_name, relative_dir_name
            )
            copy_file_info['install_path'] = os.path.join(
                file_info.get('install_path', ''), target_name, relative_dir_name
            )
            file_info_list.append(copy_file_info)

        for name in dirs_to_remove:
            dirs.remove(name)
    return file_info_list, dir_info_list


def expand_file_info_asterisk(parsed_result: FileInfoParsedResult,
                              env: ParseEnv) -> Iterator[FileInfoParsedResult]:
    """展开FileInfoParsedResult中的星号。"""
    file_info = parsed_result.file_info
    if check_contain_asterisk(file_info.get('value', '')):
        src_prefix = get_src_prefix(file_info, env)
        src_targets = sorted(glob.glob(get_src_target(file_info, env)))
        if 'exclude' in file_info:
            exclude = list(map(methodcaller('strip'), file_info['exclude'].split(';')))
        else:
            exclude = []
        for src_target in src_targets:
            value = os.path.relpath(src_target, src_prefix)
            if value in exclude:
                continue
            new_file_info = file_info.copy()
            new_file_info['value'] = value
            if 'pkg_inner_softlink' in new_file_info:
                # pkg_inner_softlink中的特殊变量$(FILE)替换为展开后的文件名
                pkg_inner_softlink = new_file_info['pkg_inner_softlink']
                new_file_info['pkg_inner_softlink'] = pkg_inner_softlink.replace(
                    '$(FILE)', os.path.basename(src_target)
                )
            yield parsed_result._replace(file_info=new_file_info)
    else:
        yield parsed_result


def trans_to_stream(item: Any) -> Iterator[Any]:
    """转换为流。"""
    yield item


def is_run_package(package_attr: PackageAttr) -> bool:
    """是否为run打包。"""
    if package_attr.get('suffix') == 'run':
        return True
    return False


def need_dereference(file_info: FileInfo) -> bool:
    """是否需要解引用。"""
    if 'dereference' in file_info:
        return True
    return False


def need_expand(file_info: FileInfo, get_src_target_func: Callable[[FileInfo], str]) -> bool:
    """是否需要展开子目录。"""
    if file_info.get('entity') == 'true':
        return False
    if file_info.get('pkg_only') == 'true':
        return False
    src_target = get_src_target_func(file_info)
    if os.path.isdir(src_target):
        if need_dereference(file_info):
            return True
        if os.path.islink(src_target):
            return False
        return True
    return False


def expand_file_info(parsed_result: FileInfoParsedResult,
                     use_move: bool,
                     get_src_target_func: Callable[[FileInfo], str]) -> FileInfoParsedResult:
    """展开FileInfoParsedResult中的目录。"""
    file_info = parsed_result.file_info
    if need_expand(file_info, get_src_target_func):
        # 如果当前是文件夹，需要展开计算
        expand_infos, dir_infos = expand_dir(file_info, get_src_target_func)
        # 实测发现，对于opp包，整体目录cp的安装速度要快于目录中各文件mv
        # 可能的原因是，cp遍历目录的速度较快，并且目录中的文件都比较小。mv依赖shell迭代目录中的所有文件。
        return FileInfoParsedResult(
            merge_dict(file_info, {'is_dir': True}), [], dir_infos, expand_infos
        )

    if use_move:
        return FileInfoParsedResult(
            None, [file_info], parsed_result.dir_infos, parsed_result.expand_infos
        )

    return parsed_result


def trans_file_info_to_result(file_info: FileInfo) -> FileInfoParsedResult:
    """file_info转换为FileInfoParsedResult。"""
    return FileInfoParsedResult(file_info, [], [], [])


def query_file_feature(parsed_result: FileInfoParsedResult,
                       env: ParseEnv):
    """查询文件feature"""
    file_info = parsed_result.file_info
    if file_info.get("is_query_feature") == "true" and env.query_feature:
        src_target = get_src_target(file_info, env)
        if len(inspect.signature(env.query_feature).parameters) == 1:
            feature_list = env.query_feature(src_target)
        else:
            feature_list = env.query_feature(src_target, os.path.join(env.delivery_dir, 'lib', 'host'))
        file_info["feature"] = set(feature_list)
        
    return parsed_result


def parse_file_element(file_ele: ET.Element,
                       file_config: Dict[str, str],
                       loaded_block: LoadedBlockElement,
                       package_attr: PackageAttr,
                       env: ParseEnv) -> Iterator[FileInfoParsedResult]:
    """解析file元素。"""
    file_info = merge_dict(file_config, file_ele.attrib)
    file_info = evaluate_info(file_info, loaded_block, env.env_dict)

    if not env.need_package_func(file_info):
        return

    if package_attr.get('expand_asterisk', False):
        expand_asterisk_func = partial(expand_file_info_asterisk, env=env)
    else:
        expand_asterisk_func = trans_to_stream

    if is_run_package(package_attr):
        if 'install_path' not in file_info:
            file_info['install_path'] = ''

    trans_file_info_func = pipe(
        trans_file_info_to_result,
        expand_asterisk_func,
        partial(map, partial(config_hash, env=env)),
        partial(
            map,
            partial(
                expand_file_info,
                use_move=loaded_block.use_move,
                get_src_target_func=partial(get_src_target, env=env)
            )
        ),
    )

    yield from trans_file_info_func(file_info)


def parse_file_info_elements(loaded_block: LoadedBlockElement,
                             default_config: Dict[str, str],
                             package_attr: PackageAttr,
                             env: ParseEnv) -> Iterator[FileInfoParsedResult]:
    """解析file_info元素。"""
    file_info_eles: List[ET.Element] = loaded_block.root_ele.findall('file_info')
    for file_info_ele in file_info_eles:
        file_config = merge_dict(
            default_config,
            file_info_ele.attrib,
            {'module': file_info_ele.attrib.get('value')}
        )

        for sub_item in list(file_info_ele):
            yield from parse_file_element(
                sub_item, file_config, loaded_block, package_attr, env
            )


def parse_op_file_elements(loaded_block: LoadedBlockElement,
                           default_config: Dict[str, str],
                           env: ParseEnv) -> Iterator[FileInfoParsedResult]:
    """解析op_file元素。"""
    op_file_eles: List[ET.Element] = loaded_block.root_ele.findall('op_file') 

    for op_file_ele in op_file_eles:
        file_config = merge_dict(
            default_config,
            op_file_ele.attrib,
            {'module': op_file_ele.attrib.get('value')}
        )
        json_path = file_config.get('json_path')
        if json_path is None:
            raise PackageConfigError("op_file's 'json_path' attr not set")
        json_path = os.path.join(env.delivery_dir, file_config.get('json_path'))
        yield from parse_autogen_op_files_json_config(file_config, json_path, loaded_block, env)


def parse_autogen_op_files_json_config(file_config, 
                                       json_path: str,
                                       loaded_block: LoadedBlockElement,
                                       env: ParseEnv):
    """解析autogen_op_files.json元素。"""
    with open(json_path, encoding='utf-8') as file:
        json_dict = json.load(file)

    op_list = json_dict.get("op_list", {})
    for op_name in sorted(op_list):
        feature_list = op_list[op_name].get("feature", [])
        files_list = op_list[op_name].get("files", [])

        for filepath in sorted(files_list):
            file_info = file_config.copy()
            file_info['value'] = filepath
            file_info['feature'] = set(feature_list)
            file_info['install_path'] = os.path.join(
                file_info['install_path'], os.path.dirname(filepath)
            )
            src_path = get_src_target(file_info, env)
            if os.path.isdir(src_path):
                file_info['entity'] = 'true'
            file_info = evaluate_info(file_info, loaded_block, env.env_dict)
            yield trans_file_info_to_result(file_info)


def get_path_infos(pkg_ele: ET.Element,
                   default_config: Dict[str, str],
                   loaded_block: LoadedBlockElement,
                   env: ParseEnv) -> List[Dict[str, str]]:
    """获取路径信息。"""
    return invoke(
        pipe(
            list,
            partial(map, attrgetter('attrib')),
            partial(map, partial(merge_dict, default_config)),
            partial(
                map,
                partial(evaluate_info, loaded_block=loaded_block, env_dict=env.env_dict, pkg=True)
            ),
            partial(
                filter,
                env.need_package_func
            )
        ),
        pkg_ele
    )


def get_path_infos_by_eles(pkg_eles: List[ET.Element],
                           default_config: Dict[str, str],
                           loaded_block: LoadedBlockElement,
                           env: ParseEnv) -> List[Dict[str, str]]:
    """根据节点列表获取路径信息。"""
    return flatten([
        get_path_infos(pkg_ele, default_config, loaded_block, env)
        for pkg_ele in pkg_eles
    ])


def parse_pkg_mods(path_infos: List[Dict[str, str]]) -> List[PkgMod]:
    """解析pkg元素。"""
    return [
        PkgMod(
            path_info['value'],
            int(path_info['pkg_mod'], 8)
        )
        for path_info in path_infos
    ]


def parse_pkg_softlinks(path_infos: List[Dict[str, str]]) -> List[PkgMod]:
    """解析pkg_softlink元素。"""
    return [
        PkgSoftlink(
            dst_path=path_info['value'],
            src_path=path_info['src_path'],
        )
        for path_info in path_infos
    ]


def parse_paths_element(root_ele: ET.Element,
                        tag_name: str,
                        ex: PackageError,
                        parse_func: Callable[[List[ET.Element]], List]) -> List:
    """解析路径列表元素。"""
    tag_eles = root_ele.findall(tag_name)
    if len(tag_eles) > 1:
        raise ex

    return parse_func(tag_eles)


def unique_infos(infos: Iterable) -> List[Dict[str, str]]:
    """infos去重。"""
    cache: Set[str] = set()
    new_infos = []
    for info in infos:
        if info['value'] in cache:
            continue
        cache.add(info['value'])
        new_infos.append(info)

    return new_infos


def parse_block_config(loaded_block: LoadedBlockElement,
                       package_attr: PackageAttr,
                       parse_env: ParseEnv):
    """解析块配置。"""
    default_config = copy.copy(loaded_block.attrs)
    default_config.update(loaded_block.root_ele.attrib)

    dir_infos = parse_dir_info_elements(
        loaded_block,
        default_config,
        package_attr,
        parse_env,
    )
    file_info_results = list(
        chain(
            parse_file_info_elements(
                loaded_block,
                default_config,
                package_attr,
                parse_env,
            ),
            parse_op_file_elements(
                loaded_block,
                default_config,
                parse_env,
            ),
        )
    )

    get_path_infos_func = partial(
        get_path_infos_by_eles,
        default_config=default_config,
        loaded_block=loaded_block,
        env=parse_env
    )

    parse_pkg_mods_func = pipe(get_path_infos_func, parse_pkg_mods)
    pkg_mods = parse_paths_element(
        loaded_block.root_ele,
        'pkg_mod',
        MultiPkgModError(),
        # tar.gz才处理pkg_mod元素
        lambda x: parse_pkg_mods_func(x) if package_attr.get('suffix') == 'tar.gz' else [],
    )

    parse_pkg_softlinks_func = pipe(get_path_infos_func, parse_pkg_softlinks)
    pkg_softlinks = parse_paths_element(
        loaded_block.root_ele,
        'pkg_softlink',
        MultiPkgSoftlinkError(),
        parse_pkg_softlinks_func,
    )

    generate_infos = parse_generate_infos_by_loaded_block(
        loaded_block, default_config, parse_env.env_dict, parse_env.need_package_func
    )

    spec_infos = parse_spec_infos(loaded_block)

    return BlockConfig(
        unique_infos(
            itertools.chain(dir_infos,
                flatten(result.dir_infos for result in file_info_results)
            )
        ),
        list(flatten(map(attrgetter('move_infos'), file_info_results))),
        list(flatten(map(attrgetter('expand_infos'), file_info_results))),
        [result.file_info for result in file_info_results if result.file_info],
        pkg_mods,
        pkg_softlinks,
        generate_infos,
        spec_infos,
    )


def make_loaded_block_element(root_ele: ET.Element,
                              dst_path: str = None) -> LoadedBlockElement:
    """创建加载后的块配置。"""
    if not dst_path:
        new_dst_path = ''
    else:
        new_dst_path = dst_path

    return LoadedBlockElement(root_ele, False, new_dst_path, set(), set(), set(), {})


def parse_block_element(block_ele: ET.Element,
                        block_info_attr: Dict[str, str]) -> BlockElement:
    """解析单个块配置。"""

    def filter_attrs(attrs: Dict[str, str]) -> Dict[str, str]:
        # block属性中过滤掉dst_path与block_conf_path
        # dst_path由单独的参数传递
        # block中不需要block_conf_path
        return {
            key: value
            for key, value in attrs.items()
            if key not in ('dst_path', 'block_conf_path')
        }

    def with_merged_attrs(attrs: Dict[str, str]) -> BlockElement:
        name = attrs.get('name')
        block_conf_path = attrs.get('block_conf_path')

        if not name:
            raise BlockConfigError("block's name is not set!")

        if not block_conf_path:
            raise BlockConfigError("block's conf_path is not set!")

        return BlockElement(
            name=name,
            block_conf_path=block_conf_path,
            dst_path=attrs.get('dst_path', ''),
            chips=config_feature_to_set(attrs.get('chip'), 'chip'),
            features=config_feature_to_set(attrs.get('feature'), 'feature'),
            pkg_features=config_feature_to_set(attrs.get('pkg_feature'), 'pkg_feature'),
            attrs=filter_attrs(attrs)
        )

    return with_merged_attrs(merge_dict(block_info_attr, block_ele.attrib))


def expand_dependencies_bfs(dep_map: Dict[str, List[ET.Element]],
                            name: str) -> List[ET.Element]:
    """展开递归依赖块。"""
    visited = set((name,))
    dependencies = []

    search_list = dep_map[name]
    while search_list:
        dep_ele = search_list.pop(0)
        if dep_ele.get('name') in visited:
            continue
        visited.add(dep_ele.get('name'))
        dependencies.append(dep_ele)
        search_list.extend(dep_map.get(dep_ele.get('name'), []))

    return dependencies


def parse_dependtree(filepath: Path) -> Dict[str, List[ET.Element]]:
    """
    解析块依赖关系,生成map;
    "key": 被依赖块;"value": 依赖块
    """
    dependtree_map = {}
    if not filepath.exists():
        COMM_LOG.cilog_warning(THIS_FILE_NAME, "%s not exists!", filepath)
        return dependtree_map

    dependtree_ele = ET.parse(str(filepath)).getroot()
    block_items = dependtree_ele.findall('Block')
    if not block_items:
        raise BlockConfigError(f"no 'Block' label in {filepath}")

    for item in block_items:
        block_name = item.attrib.get('name', None)
        if block_name is None:
            raise BlockConfigError(f"block's 'name' attr not set in {filepath}")
        dependtree_map.setdefault(block_name, [])

        dependencies = item.find("Dependencies")
        if dependencies is None:
            raise BlockConfigError(f"no 'Dependencies' label in {filepath}")

        dependtree_map[block_name].extend(list(dependencies))

    # 递归解析依赖块
    tmp_map = copy.deepcopy(dependtree_map)
    for name in tmp_map.keys():
        new_value = expand_dependencies_bfs(dependtree_map, name)
        tmp_map[name] = new_value
    dependtree_map.update(tmp_map)

    return dependtree_map


def get_dependtree_filepath() -> Path:
    """获取依赖树文件路径。"""
    return Path(pkg_utils.TOP_SOURCE_DIR) / BLOCK_DEPENDTREE_CONF


def expand_blocks(block_list: List[ET.Element]) -> List[ET.Element]:
    """展开块的依赖块信息。"""
    depend_dict = parse_dependtree(get_dependtree_filepath())
    block_names = {i.attrib.get("name") for i in block_list}
    new_block_list = block_list.copy()

    # 获取依赖块信息，并更新依赖块属性
    for item in block_list:
        depend_block = depend_dict.get(item.attrib.get("name"), None)
        if not depend_block:
            continue
        for block_dep in depend_block:
            block_dep_name = block_dep.attrib.get('name', None)
            if block_dep_name is None:
                raise BlockConfigError(
                    "block_dep 'name' not set in {}".format(item.attrib.get("name")))
            if block_dep_name in block_names:
                continue
            block_names.add(block_dep_name)
            tmp_item_attr = item.attrib.copy()
            tmp_item_attr.update(block_dep.attrib)
            block_dep.attrib.update(tmp_item_attr)
            new_block_list.append(block_dep)

    return new_block_list


def parse_block_info(block_info: ET.Element) -> List[BlockElement]:
    """解析块配置。"""

    def dependtree_flag() -> str:
        return block_info.attrib.get('dependtree', 'true').lower()

    def parse_block_eles(block_eles: List[ET.Element]) -> List[BlockElement]:
        return [
            parse_block_element(block_ele, block_info.attrib) for block_ele in block_eles
        ]

    if dependtree_flag() != 'false':
        return parse_block_eles(expand_blocks(list(block_info)))

    return parse_block_eles(list(block_info))


def get_block_filepath(block_element: BlockElement) -> str:
    """获取块配置路径。"""
    return os.path.join(
        pkg_utils.TOP_SOURCE_DIR, BLOCK_CONFIG_PATH, block_element.block_conf_path,
        f'{block_element.name}.xml'
    )


def load_block_element(package_attr: PackageAttr,
                       block_element: BlockElement) -> LoadedBlockElement:
    """加载块配置。"""
    def with_filepath(block_xml: str):
        if not os.path.exists(block_xml):
            raise BlockConfigError(f"block's config xml {block_xml} does not exist!")

        try:
            return LoadedBlockElement(
                root_ele=ET.parse(block_xml).getroot(),
                use_move=package_attr.get('use_move', False),
                **{
                    name: getattr(block_element, name)
                    for name in BLOCK_ELEMENT_PASS_THROUGH_ARGS
                }
            )
        except Exception:
            raise BlockConfigError(f"dependent block configuration {block_xml} parse failed!")

    return with_filepath(get_block_filepath(block_element))


def parse_blocks(root_ele: ET.Element,
                 package_attr: PackageAttr,
                 parse_env: ParseEnv) -> List[BlockConfig]:
    """解析块列表。"""
    return [
        parse_block_config(
            loaded_block, package_attr, parse_env
        )
        for loaded_block in itertools.chain(
            [make_loaded_block_element(root_ele)],
            map(
                partial(load_block_element, package_attr),
                chain.from_iterable(
                    map(parse_block_info, root_ele.findall("block_info"))
                )
            )
        )
    ]


def get_feature_list_filepath(top_dir: str,
                              pkg_name: str,
                              chip_scenes: str,
                              feature_list: str) -> Optional[str]:
    """获取feature.list文件路径。"""
    if not feature_list:
        return None

    return os.path.join(
        top_dir, CONFIG_SCRIPT_PATH, pkg_name, chip_scenes, feature_list
    )


def parse_feature(xml_root: ET.Element, args: Namespace, top_dir: str) -> PkgFeature:
    """解析特性。"""
    feature_attr = parse_feature_config_by_root(xml_root)
    feature_list = get_feature_list(args, feature_attr)
    feature_list_path = get_feature_list_filepath(
        top_dir, args.pkg_name, args.chip_scenes, feature_list
    )
    feature = make_pkg_feature(
        combine_feature_and_feature_list(
            args.feature,
            feature_list_path
        ),
        args.feature_exclude_all
    )
    return feature


def import_function_from_script(script_path, function_name):
    """通过脚本路径和脚本中函数名，来导入需要的函数"""
    spec = importlib.util.spec_from_file_location("module_name", script_path)
    if spec:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, function_name)
        return function
    return ''


def fill_args(args: Namespace) -> Namespace:
    """填充args中必要的参数。"""
    new_args = copy.copy(args)
    if not hasattr(new_args, 'disable_multi_version'):
        new_args.disable_multi_version = None
    if not hasattr(new_args, 'version_dir'):
        new_args.version_dir = None
    return new_args


def parse_xml_config(filepath: str,
                     xml_relpath: str,
                     delivery_dir: str,
                     parse_option: ParseOption,
                     args: Namespace) -> XmlConfig:
    """解析打包xml配置。"""
    args = fill_args(args)
    try:
        tree = ET.parse(filepath)
        xml_root = tree.getroot()
    except ET.ParseError as ex:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "xmlparse %s failed: %s!", filepath, ex)
        sys.exit(FAIL)

    default_config = xml_root.attrib.copy()

    package_attr = parse_package_attr(xml_root, args)
    feature = parse_feature(xml_root, args, pkg_utils.TOP_SOURCE_DIR)
    version, version_xml = parse_version_by_package_info_and_option(
        parse_option, package_attr, pkg_utils.TOP_SOURCE_DIR
    )
    version_dir = get_version_dir(version_xml, args.disable_multi_version, args.version_dir)
    timestamp = get_timestamp(args)
    try:
        env_dict = parse_env_dict(
            parse_option.os_arch, package_attr, version, version_dir, timestamp,
            get_compat_ver_by_top_dir(version, pkg_utils.TOP_SOURCE_DIR)
        )
    except ParseOsArchError:
        COMM_LOG.cilog_error(
            THIS_FILE_NAME, "os_arch %s is not correctly configured: %s!",
            parse_option.os_arch, xml_relpath
        )
        sys.exit(FAIL)
    except IllegalVersionDir as ex:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "Illegal version_dir %s!", str(ex))
        sys.exit(FAIL)
    err_msgs, version_info = parse_version_info_by_root(
        xml_root, env_dict, version, version_xml,
        get_base_version(args, package_attr),
        get_spc_version(args, package_attr),
        timestamp
    )
    if err_msgs:
        for err_msg in err_msgs:
            COMM_LOG.cilog_error(THIS_FILE_NAME, err_msg)
        sys.exit(FAIL)

    query_feature = import_function_from_script(
        os.path.join(pkg_utils.TOP_SOURCE_DIR, package_attr.get('query_feature_script', '')), 
        "query_feature"
        )

    need_package_func = partial(
        need_package, input_feature=feature, parse_option=parse_option, env_dict=env_dict
    )

    parse_env = ParseEnv(
        need_package_func, env_dict, parse_option, delivery_dir, pkg_utils.TOP_SOURCE_DIR,
        query_feature
    )

    try:
        blocks = parse_blocks(
            xml_root, package_attr, parse_env
        )
    except MultiPkgModError:
        COMM_LOG.cilog_error(
            THIS_FILE_NAME, "only support one pkg_mod in %s!", xml_relpath
        )
        sys.exit(FAIL)

    if is_multi_version(version_dir):
        fill_is_common_path_func = partial(
            fill_is_common_path, target_env=env_dict.get('TARGET_ENV')
        )
    else:
        fill_is_common_path_func = iter

    return XmlConfig(
        xml_relpath, default_config, package_attr, version_info, blocks, version, version_xml,
        PackerConfig(fill_is_common_path_func)
    )
