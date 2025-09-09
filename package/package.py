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

from argparse import Namespace
from cProfile import Profile
from collections import namedtuple
from datetime import datetime
from functools import partial
from itertools import chain
from pstats import Stats
from typing import Dict, Iterator, List, Optional, Set, Tuple
import json
import os
import sys
import subprocess
import argparse
import yaml
import traceback
import csv
import copy

import pkg_utils
from copy_scripts import copy_and_generate_scripts
from filelist import (
    FileItem, FileList, check_filelist, create_fileitem, generate_filelist,
    get_transform_nested_path_func,
)
from packer import (
    PackageName, check_limit_size, chmod_before_package, create_makeself_pkg_params_factory,
    create_rar_package_command, create_rpm_package_command, create_run_package_command,
    create_spec_command, exec_pack_cmd, gen_spec_command, pack_sph_rpm, pkg_cmds_to_str,
    softlink_before_package, tar_package_commands, add_rpm_dir_attr
)
from pkg_parser import (
    ParseOption, XmlConfig, parse_xml_config, get_cann_version_info
)
from pkg_utils import (
    COMM_LOG, CONFIG_SCRIPT_PATH, CompressError, ContainAsteriskError, DELIVERY_PATH, FAIL,
    FilelistError, GenerateFilelistError, PackageNameEmptyError, SUCC, TOP_DIR,
    USELESS_FILES, UnknownOperateTypeError, path_join
)
from utils.funcbase import invoke, pipe, transpose
from version_info import VersionInfo, VersionInfoFile, get_version_dir

THIS_FILE_NAME = __file__

SUPPORT_TYPE_LIST = ['repack', ]


class TimeRecord:
    """时间戳记录。
    
    记录打包各阶段时间戳。"""

    def __init__(self):
        self.start_pkg = None  # 开始打包
        self.end_depend_check = None  # 结束依赖检查
        self.end_parse = None  # 结束配置解析
        self.end_generate = None  # 结束filelist生成
        self.end_pkg = None  # 结束打包


def get_comments(package_name: PackageName) -> str:
    """获取run包注释。"""
    comments = '_'.join(
        [package_name.chip_name.upper(), package_name.func_name.upper(), 'RUN_PACKAGE']
    )
    return f'"{comments}"'


def do_compress(delivery_dir: str,
                source_target: str,
                args: Namespace,
                xmlconfig: XmlConfig) -> str:
    """根据package信息压缩对应软件包。"""
    if not os.path.exists(delivery_dir):
        os.makedirs(delivery_dir, exist_ok=True)

    source_target = os.path.relpath(source_target, delivery_dir)
    suffix = xmlconfig.package_attr.get('suffix')

    if suffix == "tar.gz":
        package_name = PackageName(xmlconfig.package_attr, args, xmlconfig.version)
        pack_cmd = pkg_cmds_to_str(
            tar_package_commands(package_name, source_target, xmlconfig.package_attr, 'gzip')
        )
        exec_func = partial(exec_pack_cmd, delivery_dir, pack_cmd, package_name.getvalue())
    elif suffix == "tar.xz":
        package_name = PackageName(xmlconfig.package_attr, args, xmlconfig.version)
        pack_cmd = pkg_cmds_to_str(
            tar_package_commands(package_name, source_target, xmlconfig.package_attr, 'xz -T0')
        )
        exec_func = partial(exec_pack_cmd, delivery_dir, pack_cmd, package_name.getvalue())
    elif suffix == "rar":
        package_name = PackageName(xmlconfig.package_attr, args, xmlconfig.version)
        pack_cmd, _ = create_rar_package_command(package_name.getvalue(), source_target)
        exec_func = partial(exec_pack_cmd, delivery_dir, pack_cmd, package_name.getvalue())
    elif xmlconfig.package_attr.get('pkg_type') == 'sph.rpm':
        without_ext = os.path.splitext(xmlconfig.xml_relpath)[0]
        spec_path = os.path.join(pkg_utils.TOP_SOURCE_DIR, f'{without_ext}.spec')
        exec_func = partial(pack_sph_rpm, delivery_dir, spec_path)
    elif suffix == "rpm":
        package_name = PackageName(xmlconfig.package_attr, args, xmlconfig.version)
        pack_cmd = create_rpm_package_command(package_name, args.pkg_name)
        exec_func = partial(exec_pack_cmd, delivery_dir, pack_cmd, package_name.getvalue())
    elif suffix == "run":
        package_name = PackageName(xmlconfig.package_attr, args, xmlconfig.version)
        factory = create_makeself_pkg_params_factory(
            source_target, package_name.getvalue(), get_comments(package_name)
        )
        params = factory(
            pkg_utils.TOP_SOURCE_DIR, delivery_dir, xmlconfig.package_attr
        )
        pack_cmd, err_msg = create_run_package_command(delivery_dir, params)
        if err_msg:
            COMM_LOG.cilog_error(THIS_FILE_NAME, err_msg)
            COMM_LOG.cilog_error(THIS_FILE_NAME, "create_run_command failed!")
    else:
        COMM_LOG.cilog_error(
            THIS_FILE_NAME, "the repack type '%s' is not support!", suffix
        )
        raise CompressError(None)
    try:
        with open("build/makeself.txt", 'w') as f:
            f.write(pack_cmd)
        print(f"Filelist generated at")
    except Exception as e:
        print(f"Error writing file: {str(e)}", file=sys.stderr)
        sys.exit(1)
    return package_name.getvalue()


def make_parse_option(args_: argparse.Namespace) -> ParseOption:
    """创建解析参数。"""

    return ParseOption(
        args_.os_arch, args_.pkg_version,
        args_.build_type,
        args_.package_check,
        args_.ext_name
    )


PrivatePackageOption = namedtuple(
    'PrivatePackageOption',
    [
        'os_arch', 'package_suffix', 'not_in_name', 'pkg_version', 'ext_name',
        'chip_name', 'func_name', 'version_dir', 'disable_multi_version', 'suffix'
    ]
)


class PackageOption(PrivatePackageOption):
    """打包配置参数。"""

    def __new__(cls,
                os_arch: Optional[str] = None,
                package_suffix: Optional[str] = None,
                not_in_name: str = '',
                pkg_version: Optional[str] = None,
                ext_name: str = '',
                chip_name: Optional[str] = None,
                func_name: Optional[str] = None,
                version_dir: Optional[str] = None,
                disable_multi_version: bool = False,
                suffix: Optional[str] = None):

        return PrivatePackageOption.__new__(
            cls, os_arch, package_suffix, not_in_name, pkg_version, ext_name,
            chip_name, func_name, version_dir, disable_multi_version, suffix
        )


def generate_version_info(version_info: VersionInfo,
                          package_name: PackageName,
                          package_option: PackageOption,
                          target_path: str):
    """生成版本信息。"""
    if version_info.version_xml:
        requires = version_info.version_xml.collect_requires(package_name.func_name)
    else:
        requires = []

    itf_version_info = '\n'.join(version_info.itf_versions)
    version_dir = get_version_dir(
        version_info.version_xml, package_option.disable_multi_version, package_option.version_dir
    )

    version_info_file = VersionInfoFile(
        version_info.version, version_info.base_version, version_info.spc_version, itf_version_info, requires,
        version_dir=version_dir, timestamp=version_info.timestamp
    )
    version_info_file.save(
        os.path.join(target_path, version_info.dst_path, "version.info")
    )


def do_copy(target_conf={},
            delivery_dir='',
            release_dir='',
            config_relpath=None,
            package_name=None):
    '''
    功能描述: 根据拷贝类型来执行文件或目录拷贝
    返回值: SUCC/FAIL
    '''
    copy_type = target_conf.get('copy_type')
    if copy_type == 'delivery':
        src_target = os.path.join(
            delivery_dir,
            target_conf['src_path'],
            target_conf.get('value')
        )
    elif copy_type == 'source':
        src_target = os.path.join(
            pkg_utils.TOP_SOURCE_DIR,
            target_conf['src_path'],
            target_conf.get('value')
        )
    else:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "copy_type %s is not support for src_target %s!", copy_type, src_target)
        return FAIL

    target_name = get_target_name(target_conf)
    dst_path = os.path.join(release_dir, target_conf.get('dst_path', ''))
    pkg_mod = target_conf.get('pkg_mod', '')
    rename = target_conf.get('rename')

    cmd = ''
    if package_name.suffix == "rpm":
        dst_path = '$HOME/rpmbuild/SOURCES/'
    if not os.path.exists(dst_path):
        cmd += ' '.join(['mkdir', '-p', dst_path, '&&'])
    dst_fullpath = os.path.join(dst_path, target_name)
    if rename:
        dst_path = os.path.join(dst_path, rename)
    if (package_name.ext_name == 'sea' or package_name.ext_name == 'sea-release' or package_name.ext_name == 'sea-hitest' or package_name.os_arch == 'aoscore.aarch64') and package_name.func_name.lower() == 'runtime':
        src_target = src_target.replace('lib/host', 'lib/host/aos_core_libs')

    if 'rsync_exclude' in target_conf and not rename:
        rsync_exclude = target_conf['rsync_exclude']
        cmd += ' '.join(['rsync', '-r', '--exclude', f'"{rsync_exclude}"', src_target.rstrip('/'), dst_path])
    else:
        if 'dereference' in target_conf:
            dereference_flag = 'L'
        else:
            dereference_flag = ''
        cmd += ' '.join([f'cp -rf{dereference_flag}', src_target, dst_path])
    status = SUCC
    if status != SUCC:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "do_copy(%s) failed!", cmd)
        COMM_LOG.cilog_error(THIS_FILE_NAME, "%s", output)
        COMM_LOG.cilog_error(THIS_FILE_NAME, "please check package config %s!", config_relpath)
        return FAIL


    if pkg_mod:
        cmd = f'chmod -R {pkg_mod} {dst_fullpath}'
        if status != SUCC:
            COMM_LOG.cilog_error(THIS_FILE_NAME, "chmod(%s) failed! %s.", cmd, cmd)
            COMM_LOG.cilog_info(THIS_FILE_NAME, "%s", output)
            return FAIL

    pkg_softlink = target_conf.get('pkg_softlink')
    if pkg_softlink:
        rets = [
            creat_softlink(dst_fullpath, os.path.join(release_dir, link))
            for link in pkg_softlink
        ]
        if not all(rets):
            return FAIL
    return SUCC


def generate_hash_list(target_conf, hash_cfg_str, release_dir, package_name):

    target_name = get_target_name(target_conf)
    dst_path = os.path.join(release_dir, target_conf.get('dst_path', ''))

    cmd = ''
    if package_name.suffix == "rpm":
        dst_path = '$HOME/rpmbuild/SOURCES/'
    dst_fullpath = os.path.join(dst_path, target_name)
    hash_cmd  = "sha256sum " + dst_fullpath
    status, output = subprocess.getstatusoutput(hash_cmd)
    if status != SUCC:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "get_image_hash command: %s", hash_cmd)
        COMM_LOG.cilog_info(THIS_FILE_NAME, "get_image_hash failed!(%s)", output)
        return FAIL, None
    hash_value = output.split()[0]
    hash_cfg_str += f"{target_name}={hash_value}\n"
    return SUCC, hash_cfg_str


def generate_hash_file(delivery_dir, hash_list):
    """
    功能描述: 生成hash文件
    参数: delivery_dir打包的临时目录
          hash_list生成的cfg列表
    """
    hash_path = os.path.join(delivery_dir, "bin_hash.cfg")
    if not os.path.exists(hash_path):
        cmd = "touch " + hash_path
        status, output = subprocess.getstatusoutput(cmd)
        if status != SUCC:
            COMM_LOG.cilog_error(THIS_FILE_NAME, "%s failed!", cmd)
            COMM_LOG.cilog_info(THIS_FILE_NAME, "%s", output)
            return FAIL, None
    with open(os.path.join(hash_path),"w") as fw:
        fw.write(hash_list)
    return SUCC


def creat_softlink(source, target) -> bool:
    '''
    功能描述: 创建软连接
    参数:  source, target
    返回值: 成功或失败
    '''
    source = os.path.abspath(source.strip())
    target = os.path.abspath(target.strip())

    link_target_path = os.path.dirname(target)
    link_target_name = os.path.basename(target)
    relative_path = os.path.relpath(source, link_target_path)
    if os.path.isfile(target):
        cmd = 'rm -f {}'.format(target)
        status, output = subprocess.getstatusoutput(cmd)
        if status != SUCC:
            COMM_LOG.cilog_error(THIS_FILE_NAME, "Error: rm -f %s failed, %s" , target, output)
            return False
    if os.path.isdir(target):
        COMM_LOG.cilog_error(THIS_FILE_NAME, "Error: %s is a directory can't add soft link", target)
        return False
    if not os.path.exists(link_target_path):
        os.makedirs(link_target_path)
    tmp_dir = os.getcwd()
    os.chdir(link_target_path)
    os.symlink(relative_path, link_target_name)
    os.chdir(tmp_dir)
    return True


def generate_info_content(target_conf, ext_name) -> Iterator[str]:
    """生成info内容。"""
    def toolchain_llvm_config() -> Iterator[Tuple[str, str]]:
        if 'llvm' in ext_name:
            yield 'toolchain', 'llvm'

    content_list = [
        f'{key}={value}'
        for key, value in chain(
            target_conf['content'].items(), toolchain_llvm_config()
        )
    ]
    return content_list


def generate_version_header_content(target_conf) -> Iterator[str]:
    """生成version_header内容。"""
    guard_name = target_conf['value'].replace('.', '_').upper()
    yield f'#ifndef {guard_name}'
    yield f'#define {guard_name}'
    yield ''
    for name, value in target_conf['content'].items():
        if name.endswith('_VERSION'):
            version_infos = get_cann_version_info(name, value)
            for version_name, version_value in version_infos:
                yield f'#define {version_name} {version_value}'
        else:
            yield f'#define {name} {value}'
    yield ''
    yield f'#endif /* {guard_name} */'
    yield ''


def generate_customized_file(target_conf, release_dir, ext_name):

    dst_path = os.path.join(release_dir, target_conf.get('dst_path', ''))
    filepath = os.path.join(dst_path, target_conf.get('value'))

    generator = target_conf.get('generator', 'info')
    if generator == 'version_header':
        content_list = generate_version_header_content(target_conf)
    else:
        content_list = generate_info_content(target_conf, ext_name)

    file_content = '\n'.join(content_list)
    try:
        os.makedirs(dst_path, exist_ok=True)
        with open(filepath, 'w') as file:
            file.write(file_content)
    except Exception as ex:
        COMM_LOG.cilog_error(
            THIS_FILE_NAME, f"generate customized file {filepath} failed: {ex}!"
        )
        return FAIL

    return SUCC


def classify_package(target_config) -> int:
    """分类包文件。"""
    return 0


def classify_spc(target_config) -> int:
    """分类补丁文件。"""
    install_path = target_config.get('install_path', '')
    install_path_list = install_path.split('/')
    if not install_path_list:
        return 1
    if install_path_list[0] == 'spc':
        return 0
    return 1


def get_module(target_config) -> str:
    """获取配置模块。"""
    module = target_config.get('module', 'NA')
    return module if module else 'NA'


def get_operation(operation, target_config) -> str:
    """获取操作类型。"""
    if operation in ('copy', 'move') and target_config.get('entity') == 'true':
        return 'copy_entity'
    return operation


def get_permission(target_config) -> str:
    """获取配置权限。"""
    return target_config.get('install_mod', 'NA')


def get_owner_group(target_config) -> str:
    """获取配置属主。"""
    # install_own的可能值为$username:$usergroup
    # 防止变量在install_common_parser.sh中，被eval展开，添加\转义$
    # 由于awk会消耗1个\，所以需要2个转义符
    return target_config.get('install_own', 'NA').replace('$', '\\\\$')


def get_install_type(target_config) -> str:
    """获取安装类型。"""
    return target_config.get('install_type', 'NA')


def get_softlink(target_config) -> List[str]:
    """获取配置软链。"""
    softlink_str = target_config.get('install_softlink')
    if not softlink_str:
        return []
    return softlink_str.split(';')


def get_feature(target_config) -> Set[str]:
    """获取配置特性。"""
    return target_config['feature']


def get_chip(target_config) -> Set[str]:
    """获取配置芯片。"""
    return target_config['chip']


def get_configurable(target_config) -> str:
    """获取配置是否为配置文件。"""
    return target_config.get('configurable', 'FALSE')


def get_hash_value(target_config) -> str:
    """获取配置哈希值。"""
    return target_config.get('hash', 'NA')


def get_block(target_config) -> str:
    """获取配置块信息。"""
    return target_config.get('name', 'NA')


def get_pkg_inner_softlink(target_config) -> List[str]:
    """获取配置包内软链。"""
    softlink_str = target_config.get('pkg_inner_softlink')
    if not softlink_str:
        return []
    return softlink_str.split(';')


def create_version_info_install_info(version_info_attrib: Dict, block: str) -> FileItem:
    """创建version.info文件的安装信息。"""
    module = 'NA'
    operation = 'copy'
    target_name = 'version.info'
    relative_install_path = os.path.join(version_info_attrib.get('install_path', ''), target_name)
    is_in_docker = 'TRUE'
    install_type = 'all'

    return create_fileitem(
        module,
        operation,
        target_name,
        relative_install_path,
        is_in_docker,
        get_permission(version_info_attrib),
        get_owner_group(version_info_attrib),
        install_type,
        get_softlink(version_info_attrib),
        get_feature(version_info_attrib),
        'N',
        get_configurable(version_info_attrib),
        get_hash_value(version_info_attrib),
        block,
        get_pkg_inner_softlink(version_info_attrib),
        get_chip(version_info_attrib),
        False,
    )


def parse_install_info(infos: List,
                       operate_type,
                       filter_key,
                       classify_func=classify_package) -> Iterator[FileItem]:
    """根据配置解析生成安装信息。"""
    for target_config in infos:
        if target_config.get('pkg_only') == 'true':
            # 只打包不安装配置，跳过filelist条目生成
            continue
        target_name = get_target_name(target_config)
        if operate_type in ('copy', 'move'):
            relative_path_in_pkg = os.path.join(target_config.get('dst_path'), target_name)
            relative_install_path = path_join(target_config.get('install_path'), target_name)
            is_dir = target_config.get('is_dir', False)
        elif operate_type == 'mkdir':
            relative_path_in_pkg = 'NA'
            relative_install_path = target_config.get('value')
            is_dir = False
        elif operate_type == 'del':
            relative_path_in_pkg = 'NA'
            relative_install_path = path_join(target_config.get('install_path'), target_name)
            is_dir = False
        else:
            raise UnknownOperateTypeError(f"unknown operate type {operate_type}")

        if relative_install_path is None:
            continue

        install_type = get_install_type(target_config)
        if any(key in install_type for key in filter_key):
            is_in_docker = 'TRUE'
        else:
            is_in_docker = 'FALSE'

        fileitem = create_fileitem(
            get_module(target_config),
            get_operation(operate_type, target_config),
            relative_path_in_pkg,
            relative_install_path,
            is_in_docker,
            get_permission(target_config),
            get_owner_group(target_config),
            install_type,
            get_softlink(target_config),
            get_feature(target_config),
            'N',
            get_configurable(target_config),
            get_hash_value(target_config),
            get_block(target_config),
            get_pkg_inner_softlink(target_config),
            get_chip(target_config),
            is_dir,
        )

        classify_result = classify_func(target_config)
        yield (classify_result, fileitem)


def remove_useless_files(release_dir: str, filename: str) -> bool:
    """移除无效文件。"""
    filter_cmd = f'find "{release_dir}" -name "{filename}" | xargs rm -rf'
    status, output = subprocess.getstatusoutput(filter_cmd)
    if status != SUCC:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "execute %s failed, %s", filter_cmd, output)
        return False
    return True


def execute_repack_process(xmlconfig: XmlConfig,
                           delivery_dir: str,
                           args: Namespace,
                           package_name: PackageName = None,
                           package_option: PackageOption = None):
    """
    功能描述: 执行打包流程(拷贝--->签名--->打包)
    返回值: SUCC/FAIL
    """
    status = SUCC
    release_dir = os.path.join(
        delivery_dir, xmlconfig.default_config.get('name', 'default'))

    # 清理release_dir目录
    clean_cmd = ' '.join(["rm", "-rf", release_dir])
    if package_name.suffix == 'rpm':
        # 清理rpmbuild目录，并创建目录
        clean_cmd += ' '.join([
            ' $HOME/rpmbuild', '&&', 'mkdir', '-p',
            '$HOME/rpmbuild/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}'
        ])
    status, output = subprocess.getstatusoutput(clean_cmd)
    if status != SUCC:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "execute %s failed, %s", clean_cmd, output)
        return FAIL

    # 生成自定义文件
    for item in xmlconfig.generate_infos:
        if generate_customized_file(item, release_dir, package_option.ext_name):
            return FAIL

    if xmlconfig.package_attr.get('gen_version_info'):
        generate_version_info(xmlconfig.version_info, package_name, package_option, release_dir)

    # 设置spec文件初始变量
    if package_name.suffix == 'rpm':
        spec_param = [[],[],[],set(),0]
        for item in xmlconfig.dir_install_list:
            spec_param = add_rpm_dir_attr(item, spec_param)
    else:
        # 生成spec文件
        for item in xmlconfig.spec_infos:
            if generate_spec_info(item, xmlconfig, release_dir):
                return FAIL

    hash_cfg_str = ""
    for item in chain(xmlconfig.package_content_list, xmlconfig.move_content_list):
        if package_name.suffix == 'rpm' and 'install_path' in item:
            # 将mami.xml文件中的内容存为列表
            spec_param = gen_spec_command(item, spec_param)
        if get_target_name(item) == "bin_hash.cfg":
            ret = generate_hash_file(delivery_dir, hash_cfg_str)
            if ret != SUCC:
                COMM_LOG.cilog_error(THIS_FILE_NAME, "generate hash file failed!")
                return FAIL
        # 拷贝文件
        if do_copy(item,
                   delivery_dir,
                   release_dir,
                   xmlconfig.xml_relpath,
                   package_name):
            status = FAIL
            continue
        if "is_hash" in item:
            ret, hash_cfg_str = generate_hash_list(item, hash_cfg_str, release_dir, package_name)
            if ret != SUCC:
                COMM_LOG.cilog_error(THIS_FILE_NAME, "generate hash command %s failed!", get_target_name(item))
                return FAIL
    if package_name.suffix == 'rpm' and xmlconfig.package_attr.get('pkg_type') != 'sph.rpm':
        # 生成spec文件，spec_file_path目录下存放了初始文件
        # build/release/config/mami/scripts/mami.spec
        spec_file_path = os.path.join(pkg_utils.TOP_SOURCE_DIR,
                                      CONFIG_SCRIPT_PATH,
                                      args.pkg_name, args.chip_scenes, 'scripts')
        create_spec_command(spec_file_path, delivery_dir, spec_param, args.pkg_name)
    if status != SUCC:
        return FAIL

    # all会产生短路效果，不是必须的
    ret = all(map(partial(remove_useless_files, release_dir), USELESS_FILES))
    if not ret:
        return FAIL

    chmod_before_package(xmlconfig.pkg_mods, release_dir)
    softlink_before_package(xmlconfig.pkg_softlinks, release_dir)

    #校验包中文件或目录大小
    if args.check_size == "True":
        limit_list, tag = processing_csv_file(
            release_dir, package_name.func_name, package_name.chip_name, args.build_type
        )
        if not tag:
            return FAIL
        if limit_list:
            abspath = os.path.abspath(release_dir)
            replace_path = abspath + "/"
            result = check_add_dir(replace_path, abspath, limit_list)
            if not result:
                return FAIL

    pkg_output_dir = os.path.join(delivery_dir, args.pkg_output_dir)
    try:
        package_name = do_compress(pkg_output_dir, release_dir, args, xmlconfig)
    except CompressError:
        return FAIL

    if not args.no_check_limit_size and xmlconfig.package_attr.get('limit_size'):
        limit_size = int(xmlconfig.package_attr.get('limit_size'))
        package_filepath = os.path.join(pkg_output_dir, package_name)
        ret, file_size = check_limit_size(package_filepath, limit_size)
        if not ret:
            COMM_LOG.cilog_error(
                THIS_FILE_NAME, "\npackage %s size: %d exceeds limit: %d!",
                package_filepath, file_size, limit_size
            )
            return FAIL

    COMM_LOG.cilog_info(THIS_FILE_NAME, "package %s successfully!", package_name)
    return status


def check_path_is_conflict(xmlconfig):
    """
    功能描述: 检查打包时安装路径与软连接路径是否冲突
    参数: xmlconfig
    返回值: SUCC/FAIL
    """
    install_path_list = set()
    pkg_softlink_list = set()
    for item in xmlconfig.package_content_list:
        value_list = item.get('value').split('/')
        target_name = value_list[-1] if value_list[-1] else value_list[-2]
        if item.get('install_path'):
            install_path_list.add(
                os.path.join(item['install_path'], target_name)
            )
        if item.get('pkg_inner_softlink'):
            pkg_softlink = item.get('pkg_inner_softlink')
            pkg_softlink_list.add(pkg_softlink)
    if install_path_list & pkg_softlink_list:
        COMM_LOG.cilog_info(THIS_FILE_NAME, 'intersection:{}'.format(install_path_list & pkg_softlink_list))
        COMM_LOG.cilog_info(THIS_FILE_NAME, 'path conflicting: pkg_inner_softlink dir equals install_path!!')
        return FAIL
    return SUCC


def checksum_value(limit_value, release_dir):
    '''
    功能描叙: 校验传入的文件或目录大小是否合格
    参数:
    limit_value: limit.csv中的一行数据如[compiler/bin, 3976, 110%]
    返回值: True/False
    '''
    path = os.path.join(release_dir, limit_value[1])
    if len(limit_value) >= 7:
        try:
            max_value = int(limit_value[4])
        except ValueError:
            COMM_LOG.cilog_error(THIS_FILE_NAME, "{0} configuration is not standard., Please check limit.csv.".format(path))
            return True
    else:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "{0} configuration is less than four, Please check limit.csv.".format(path))
        return True
    if not os.path.exists(path):
        COMM_LOG.cilog_warning(THIS_FILE_NAME, "{0} doesn't exist, Please check limit.csv.".format(path))
        return True
    size = 0
    for root, dirs, files in os.walk(path):
        size += os.path.getsize(root)
        for f in files:
            filepath = os.path.join(root, f)
            if os.path.islink(filepath):
                continue
            if not os.path.exists(filepath):
                continue
            size += os.path.getsize(os.path.join(root, f))
    if size == 0:
        size = os.path.getsize(path)
    if size > max_value * 1024:
        COMM_LOG.cilog_error(
            THIS_FILE_NAME, f"\n{path} size {size} bytes exceeds maximum {max_value * 1024} bytes"
        )
        return False
    return True


def processing_csv_file(release_dir, package_name, chip_name, build_type):
    '''
    功能描叙: 处理limit.csv文件数据
    返回值: [],True/[],False
    '''
    succ = True
    limit_list = []
    product = os.path.basename(os.path.dirname(release_dir))
    limit_path = os.path.join(pkg_utils.TOP_SOURCE_DIR, CONFIG_SCRIPT_PATH, "common/limit.csv")
    if not os.path.exists(limit_path):
        COMM_LOG.cilog_warning(THIS_FILE_NAME, "{0} doesn't exist.".format(limit_path))
        return limit_list, succ
    with open(limit_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for data in reader:
            if not data:
                COMM_LOG.cilog_warning(THIS_FILE_NAME, "The limit.csv file contains empty lines.")
                continue
            if package_name == data[0] and chip_name == data[5] and product == data[6] and build_type == data[7].lower():
                if data[1][-1] == "/":
                    limit_list.append(data[1][:-1])
                else:
                    limit_list.append(data[1])
                res = checksum_value(data, release_dir)
                if not res:
                    succ = False
    return limit_list, succ


def check_add_dir(package_path, dirs, limit_list, succ=True):
    """
    功能描述: 校验新增目录
    参数: xmlconfig, path, limit_list
    返回值: False/True
    """
    for limit_path in limit_list:
        if dirs == os.path.join(os.path.split(dirs)[0], limit_path):
            return succ
    for dir_file in os.listdir(dirs):
        path = os.path.join(dirs, dir_file)
        relative_path = path.replace(package_path, "")
        if os.path.isfile(path) and relative_path not in limit_list:
            COMM_LOG.cilog_error(THIS_FILE_NAME, "{0} is not in limit.csv file and is newly added.".format(path))
            succ = False
        elif os.path.isdir(path) and relative_path not in limit_list:
            succ = check_add_dir(package_path, path, limit_list, succ)
    return succ


def check_pkg_name(func_name, xmlconfig, not_in_name):
    '''
    功能描述: 检查包名是否合格
    参数: func_name, xmlconfig, not_in_name
    返回值: False/True
    '''
    pkg_conf_path = os.path.join(
        pkg_utils.TOP_SOURCE_DIR, CONFIG_SCRIPT_PATH, "common/pkg_conf.yml"
    )
    if not os.path.exists(pkg_conf_path):
        COMM_LOG.cilog_warning(THIS_FILE_NAME, "{0} doesn't exist.".format(pkg_conf_path))
        return True
    if not func_name:
        func_name = xmlconfig.package_attr.get("func_name")
    func_name = None if "func_name" in not_in_name else func_name
    if func_name is None:
        return True
    with open(pkg_conf_path, "r", encoding="utf-8") as file_load:
        yaml_conf = yaml.safe_load(file_load)
    pkg_name_whitelist = yaml_conf.get("PKG_NAME_WHITELIST")
    whitelist_fullname = pkg_name_whitelist.get("FULLNAME")
    whitelist_include = pkg_name_whitelist.get("INCLUDE", [])
    for include_data in whitelist_include:
        if include_data in func_name:
            return True
    if func_name not in whitelist_fullname:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "{0} package name check failed, whitelist is [{1}]".format(func_name, ", ".join(whitelist_fullname)))
        return False
    return True


def get_target_name(target_conf) -> str:
    """获取目标名。"""
    rename = target_conf.get('rename')
    if rename:
        return rename

    value_list = target_conf.get('value').split('/')
    target_name = value_list[-1] if value_list[-1] else value_list[-2]
    return target_name


def get_dst_fullpath(target_conf, target_name: str) -> str:
    """获取目的全路径。"""
    dst_path = target_conf.get('dst_path')
    return os.path.join(dst_path, target_name)


def get_install_fullpath(target_conf, target_name: str) -> Optional[str]:
    """获取安装全路径。"""
    return path_join(target_conf.get('install_path'), target_name)


def generate_swbom_package_info(package_content_list: Iterator[Dict],
                                package_name: PackageName,
                                spi_filepath: str):
    """生成swbom打包信息。"""
    copy_commands = []
    for target_conf in package_content_list:
        copy_type = target_conf.get('copy_type')
        src_target = os.path.join(
            target_conf['src_path'],
            target_conf.get('value')
        )

        target_name = get_target_name(target_conf)
        dst_fullpath = get_dst_fullpath(target_conf, target_name)

        info = {
            'copy_type': copy_type,
            'src_target': src_target,
            'dst_target': dst_fullpath,
        }

        if package_name.suffix == 'run':
            # run包添加install_target属性
            install_fullpath = get_install_fullpath(target_conf, target_name)
            info['install_target'] = install_fullpath

        copy_commands.append(info)

    spi_dir = os.path.dirname(spi_filepath)
    if not os.path.exists(spi_dir):
        os.makedirs(spi_dir, exist_ok=True)

    package_infos = {
        package_name.getvalue(): copy_commands,
    }

    with open(spi_filepath, 'w', encoding='utf-8') as file:
        json.dump(package_infos, file, sort_keys=True, indent=4)


def gen_file_install_list(xmlconfig: XmlConfig,
                          filter_key,
                          classify_func) -> Tuple[FileList, FileList]:
    """生成filelist列表。"""
    file_install_list = []
    backup_file_list = []

    dir_filelist = parse_install_info(
        xmlconfig.dir_install_list, 'mkdir', filter_key, classify_func
    )
    move_filelist = parse_install_info(
        xmlconfig.move_content_list, 'move', filter_key, classify_func
    )
    pkg_filelist = parse_install_info(
        xmlconfig.package_content_list, 'copy', filter_key, classify_func
    )
    gen_filelist = parse_install_info(
        xmlconfig.generate_infos, 'copy', filter_key, classify_func
    )
    # file_info中配置为文件夹，这里是被展开的文件,则需要单独删除
    del_filelist = parse_install_info(
        xmlconfig.expand_content_list, 'del', filter_key, classify_func
    )
    collect_filelist = tuple(chain(dir_filelist, move_filelist, pkg_filelist, gen_filelist))
    if collect_filelist:
        classifies, tmp_filelist = transpose(collect_filelist)
        tmp_filelist = tuple(xmlconfig.packer_config.fill_is_common_path(tmp_filelist))
        collect_filelist = transpose((classifies, tmp_filelist))

    all_filelist = chain(collect_filelist, del_filelist)
    for classify_result, fileitem in all_filelist:
        if classify_result == 0:
            file_install_list.append(fileitem)
        else:
            backup_file_list.append(fileitem)

    if xmlconfig.version_info.install_version_info:
        file_install_list.append(
            create_version_info_install_info(
                xmlconfig.version_info.install_version_info_attrib, xmlconfig.block_name
            )
        )

    return file_install_list, backup_file_list


def generate_filelist_file_by_xmlconfig(xmlconfig: XmlConfig,
                                        delivery_dir: str,
                                        filter_key: List[str],
                                        package_check: bool) -> List[FileList]:
    """生成文件列表文件。"""
    package_name = xmlconfig.default_config.get('name', '')
    check_move = xmlconfig.package_attr.get('use_move', False)
    transform_nested_path_func = get_transform_nested_path_func(
        xmlconfig.package_attr.get('parallel') or check_move
    )
    check_features = xmlconfig.package_attr.get('check_features', False)

    if package_name not in ('spc', ''):
        file_install_list, backup_file_list = invoke(
            pipe(
                gen_file_install_list,
                partial(map, transform_nested_path_func),
                tuple,
            ),
            xmlconfig, filter_key, classify_package
        )
        generate_filelist(file_install_list, delivery_dir, 'filelist.csv')
        # 先生成再检查，有利于问题定位
        if package_check:
            check_filelist(file_install_list, check_features, check_move)
    # 生成补丁包的安装列表文件
    elif package_name == 'spc':
        file_install_list, backup_file_list = invoke(
            pipe(
                gen_file_install_list,
                partial(map, transform_nested_path_func),
                tuple,
            ),
            xmlconfig, filter_key, classify_spc
        )
        generate_filelist(file_install_list, delivery_dir, 'filelist.csv')
        generate_filelist(backup_file_list, delivery_dir, 'filelist_spc.csv')
        if package_check:
            check_filelist(file_install_list, check_features, check_move)
    else:
        raise PackageNameEmptyError()

    return [file_install_list, backup_file_list]


def generate_spec_info(spec_info, xmlconfig, release_dir):
    """根据包配置文件中的spec_info生成spec文件"""
    delivery_dir = os.path.dirname(release_dir)
    spec_param = [[],[],[],set(),0]
    for item in xmlconfig.dir_install_list:
        spec_param = add_rpm_dir_attr(item, spec_param)
    pkg_list = copy.deepcopy(xmlconfig.package_content_list + xmlconfig.move_content_list)
    for item in pkg_list:
        if 'install_path' not in item:
            continue
        if spec_info["install_type"] == "-" + item["install_type"]:
            continue
        # 将xml文件中的内容存为列表
        spec_param = gen_spec_command(item, spec_param)
    spec_file_path = os.path.join(pkg_utils.TOP_SOURCE_DIR,
                                    CONFIG_SCRIPT_PATH,
                                    args.pkg_name, args.chip_scenes, 'scripts')
    org_spec_filename = create_spec_command(
        spec_file_path, delivery_dir, spec_param, args.pkg_name)
    dst_spec_filename = os.path.join(release_dir, spec_info["dst_path"], spec_info["value"])
    cmd = ' '.join(["mkdir -p",os.path.join(release_dir, spec_info["dst_path"]),
        "&&", "mv", org_spec_filename, dst_spec_filename])
    status, output = subprocess.getstatusoutput(cmd)
    if status != SUCC:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "gen spec file(%s) failed!", cmd)
        COMM_LOG.cilog_error(THIS_FILE_NAME, "%s", output)
        return FAIL
    return SUCC


def get_pkg_xml_relpath(args: Namespace) -> str:
    """获取包配置文件相对路径。"""
    def parts():
        yield CONFIG_SCRIPT_PATH
        yield args.pkg_name
        if args.chip_scenes:
            yield args.chip_scenes
        if args.component:
            yield args.component
        # 可以通過build_rule指定xml_file，而且优先级高于默认值
        if args.xml_file:
            yield args.xml_file
        else:
            yield f'{args.pkg_name}.xml'
    return os.path.join(*parts())


def main(product_name='',
         chip_scenes='',
         pkg_name='',
         xml_file='',
         args=None,
         time_record=None
         ):
    """
    功能描述: 执行打包流程(解析配置--->生成文件列表--->执行拷贝/打包动作)
    参数: product_name, chip_scenes, pkg_name, os_arch, type
    返回值: SUCC/FAIL
    """
    status = SUCC
    delivery_root_dir = os.path.join(TOP_DIR, DELIVERY_PATH)
    delivery_dir = os.path.join(delivery_root_dir, product_name)

    if not os.path.exists(delivery_dir):
        os.makedirs(delivery_dir)

    time_record.end_depend_check = datetime.now()

    config_relpath = get_pkg_xml_relpath(args)
    pkg_xml_file = os.path.join(pkg_utils.TOP_SOURCE_DIR, config_relpath)
    parse_option = make_parse_option(args)

    try:
        xmlconfig = parse_xml_config(
            pkg_xml_file, config_relpath, delivery_dir, parse_option, args
        )
    except ContainAsteriskError as ex:
        COMM_LOG.cilog_error(
            THIS_FILE_NAME, f"Value contain '*' in {config_relpath}. value is '{ex.value}'."
        )
        return FAIL

    # 执行包名校验

    time_record.end_parse = datetime.now()

    if pkg_name in ['driver', 'firmware']:
        filter_key = ['all', 'docker']
    elif pkg_name in ['aicpu_kernels_device', 'aicpu_kernels_host']:
        filter_key = []
    else:
        filter_key = ['all', 'run']

    # 生成安装列表文件
    try:
        generate_filelist_file_by_xmlconfig(
            xmlconfig, delivery_dir, filter_key,
            args.package_check or xmlconfig.package_attr.get('package_check')
        )
    except PackageNameEmptyError:
        COMM_LOG.cilog_error(
            THIS_FILE_NAME, f'package name is empty in {xml_file}, please check it'
        )
        return FAIL
    except GenerateFilelistError as ex:
        COMM_LOG.cilog_error(
            THIS_FILE_NAME, f'generate filelist {ex.filename} failed!'
        )
        return FAIL
    except FilelistError as ex:
        COMM_LOG.cilog_error(THIS_FILE_NAME, 'check filelist error! %s', str(ex))
        print(str(ex))

    copy_and_generate_scripts(pkg_utils.TOP_SOURCE_DIR, delivery_dir, xmlconfig.package_attr)

    package_option = PackageOption(
        args.os_arch, args.package_suffix, args.not_in_name, args.pkg_version, args.ext_name,
        chip_name=args.chip_name, func_name=args.func_name, version_dir=args.version_dir,
        disable_multi_version=args.disable_multi_version, suffix=args.suffix)

    package_name = PackageName(xmlconfig.package_attr, args, xmlconfig.version)

    swbom_filepath = os.path.join(
        delivery_root_dir, 'swbom',
        f'swbom.package_info.{pkg_name}_{package_name.chip_name.lower()}.json'
    )
    generate_swbom_package_info(
        chain(xmlconfig.package_content_list, xmlconfig.move_content_list),
        package_name, swbom_filepath
    )

    time_record.end_generate = datetime.now()

    if args.no_action:
        return SUCC

    # 检查install_path与pkg_inner_softlink路径是否冲突，若冲突则报错
    status = check_path_is_conflict(xmlconfig)
    if status == FAIL:
        return status

    # 执行拷贝/打包动作
    status = execute_repack_process(xmlconfig, delivery_dir, args,
                                    package_name=package_name, package_option=package_option)
    return status


def time_measure(*args, **kwargs):
    """打包时间度量。"""
    time_record = TimeRecord()
    kwargs['time_record'] = time_record

    time_record.start_pkg = datetime.now()

    need_profile = kwargs['args'].profile
    if need_profile:
        profiler = Profile()
        profiler.runcall(main, *args, **kwargs)
        ret = SUCC
    else:
        ret = main(*args, **kwargs)
    time_record.end_pkg = datetime.now()

    delta = time_record.end_pkg - time_record.start_pkg
    COMM_LOG.cilog_info(THIS_FILE_NAME, 'package cost total time {0}'.format(delta))

    if time_record.end_depend_check is not None:
        delta = time_record.end_depend_check - time_record.start_pkg
        COMM_LOG.cilog_info(THIS_FILE_NAME, 'depend check cost time {0}'.format(delta))

    if time_record.end_parse is not None:
        delta = time_record.end_parse - time_record.end_depend_check
        COMM_LOG.cilog_info(THIS_FILE_NAME, 'parse cost time {0}'.format(delta))

    if time_record.end_generate is not None:
        delta = time_record.end_generate - time_record.end_parse
        COMM_LOG.cilog_info(THIS_FILE_NAME, 'generate cost time {0}'.format(delta))

        delta = time_record.end_pkg - time_record.end_generate
        COMM_LOG.cilog_info(THIS_FILE_NAME, 'package cost time {0}'.format(delta))

    if need_profile:
        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats()

    return ret


def args_prase():
    """
    功能描述 : 脚本入参解析
    参数 : 调用脚本的传参
    返回值 : 解析后的参数值
    """
    parser = argparse.ArgumentParser(
        description='This script is for spiltpackage repack processing.')
    parser.add_argument('-c', '--chip_scenes', metavar='chip_scenes', required=False, dest='chip_scenes', nargs='?', const='',
                        default='', help='This parameter define chipid for package.')
    parser.add_argument('--component', metavar='component', required=False, nargs='?', const=None,
                        default=None, help='This parameter define component for package.')
    parser.add_argument('-p', '--product', metavar='product', required=False, dest='product_name', nargs='?', const='',
                        default='', help='This parameter define product name.')
    parser.add_argument('-s', '--sign', metavar='sign', required=False, dest='cms_sign', nargs='?', const='',
                        default='', help='This parameter define cms_sign true or false.')
    parser.add_argument('-n', '--pkg_name', metavar='pkg_name', required=True,
                        help='This parameter define pkg_name for config_xml.')
    parser.add_argument('-o', '--os_arch', metavar='os_arch', required=False, dest='os_arch', nargs='?', const='',
                        default=None, help="This parameter define the package's os_arch")
    parser.add_argument('-t', '--type', metavar='type', required=False, dest='type', nargs='?', const='',
                        default=SUPPORT_TYPE_LIST[0], help="This parameter define this script's function")
    parser.add_argument('-i', '--not_in_name', metavar='not_in_name', required=False, dest='not_in_name', nargs='?', const='',
                        default='', help="This parameter define the package's name not contain the element")
    parser.add_argument('-v', '--pkg_version', metavar='pkg_version', required=False, dest='pkg_version', nargs='?', const='',
                        default='', help="This parameter define the version for package.")
    parser.add_argument('-e', '--ext_name', metavar='ext_name', required=False, dest='ext_name', nargs='?', const='',
                        default='', help="This parameter define the package's ext_name")
    parser.add_argument('-a', '--atlas_pkg_check', metavar='atlas_pkg_check', required=False, dest='atlas_pkg_check', nargs='?', const='',
                        default='', help="This parameter define the package's atlas_pkg_check")
    parser.add_argument('--package_suffix', nargs='?', const='none',
                        default='none', help="This parameter define the package suffix, debug or none")
    parser.add_argument('--suffix', metavar='suffix', required=False, dest='suffix', nargs='?', const='',
                        default=None, help="This parameter define the package suffix, for example such as tar.gz")
    parser.add_argument('-b', '--build_type', metavar='build_type', required=False, dest='build_type', nargs='?', const='',
                        default='debug', help="This parameter define release type of package")
    parser.add_argument('-f', '--feature', metavar='feature', nargs='?', const='',
                        default='', help="This parameter define feature of package")
    parser.add_argument('--feature_list', metavar='feature_list', nargs='?', const=None,
                        default=None, help="Package feature list file.")
    parser.add_argument('--feature-exclude-all', action='store_true',
                        help='This parameter define exclude common files when package.')
    parser.add_argument('-x', '--xml', metavar='xml_file', required=False, dest='xml_file', nargs='?', const='',
                        default='', help="This parameter define xml file")
    parser.add_argument('--chip_name', metavar='chip_name', required=False, dest='chip_name', nargs='?', const=None,
                        default=None, help="This parameter define package chip name, has higher priority than chip name in xml")
    parser.add_argument('--func_name', metavar='func_name', required=False, dest='func_name', nargs='?', const=None,
                        default=None, help="This parameter define package func name, has higher priority than func name in xml")
    parser.add_argument('--source_root', metavar='source_root', required=False, dest='source_root', nargs='?', const='',
                        help='source root dir.')
    parser.add_argument('--version_dir', nargs='?', const='', default='', help='Set version dir.')
    parser.add_argument('--base-version', metavar='base_version', help='Set spc base version.')
    parser.add_argument('--spc-version', metavar='spc_version', nargs='?', const='', default='', help='Set spc version.')
    parser.add_argument('--tag', metavar='tag', nargs='?', const='', default='')
    parser.add_argument('--buildinfo_path', metavar='buildinfo_path', nargs='?', const='', default='', help='buildinfo config path')
    parser.add_argument('--disable-multi-version', action='store_true', help='Disable multi version.')
    parser.add_argument('--no-action', action='store_true', help="Do not do action if this option specified.")
    # 检查打包配置
    parser.add_argument('--package-check', action='store_true', help='check package config.')
    parser.add_argument('--profile', action='store_true', help="Profile program.")  # 性能剖析时将 store_true 改为 store_false
    parser.add_argument('--check_size', nargs='?', const='', default='', help="Check the size of a file or directory.")
    parser.add_argument('--no-check-limit-size', action='store_true', help="Don't check limit size.")
    parser.add_argument('--pkg-name-style', metavar='pkg_name_style', default='common', help='Package name style.')
    parser.add_argument('--pkg-output-dir', default='', help='Package output dirpath.')
    return parser.parse_args()


if __name__ == "__main__":
    COMM_LOG.cilog_info(THIS_FILE_NAME, "%s", " ".join(sys.argv))
    args = args_prase()
    try:
        if args.source_root:
            pkg_utils.TOP_SOURCE_DIR = args.source_root
        if args.build_type == '':
            args.build_type = 'debug'
        else:
            args.build_type = args.build_type.lower()
        status = time_measure(
            args.product_name, args.chip_scenes,
            args.pkg_name,
            args.xml_file, args=args
        )
    except Exception as e:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "exception is occurred (%s)!", e)
        COMM_LOG.cilog_info(THIS_FILE_NAME, "%s", traceback.format_exc())
        status = FAIL
    sys.exit(status)
