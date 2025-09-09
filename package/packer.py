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

import os
import shutil
import subprocess
from argparse import Namespace
from functools import partial
from itertools import chain
from pathlib import Path
from operator import attrgetter, methodcaller
from subprocess import PIPE, STDOUT
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

from utils import iter_text
from utils.funcbase import identity, invoke, pipe

from pkg_parser import PkgMod, PkgSoftlink
from pkg_utils import COMM_LOG, CompressError, GetSpecPropertyError, strip_lines

THIS_FILE_NAME = __file__


class PackageName:
    """包名。"""

    def __init__(self,
                 package_attr,
                 args: Namespace,
                 version: str):
        self.chip_name = args.chip_name or package_attr.get('chip_name')
        self.suffix = args.suffix or package_attr.get('suffix')
        self.func_name = get_func_name(args.func_name, package_attr)
        self.chip_plat = package_attr.get('chip_plat')
        if package_attr.get('spc_version_show_in_pkg_name') and package_attr.get('spc_version'):
            self.spc_version = package_attr['spc_version'].lower()
        else:
            self.spc_version = None

        self.deploy_type = package_attr.get('deploy_type')
        self.version = version.lower()
        self.not_in_name_list = args.not_in_name.split(",")
        self.os_arch = args.os_arch
        self.package_suffix = args.package_suffix
        self.ext_name = args.ext_name
        if args.pkg_name_style == 'underline':
            self.name_sep = '_'
        else:
            self.name_sep = '-'

    def get_attribute(self, name: str) -> Optional[str]:
        """获取属性。"""
        if name in self.not_in_name_list:
            return None
        return getattr(self, name)

    def getvalue(self) -> str:
        chip_name = self.get_attribute('chip_name')
        func_name = self.get_attribute('func_name')
        version = self.get_attribute('version')
        os_arch = self.get_attribute('os_arch')
        chip_plat = self.get_attribute('chip_plat')
        deploy_type = self.get_attribute('deploy_type')
        ext_name = self.get_attribute('ext_name')
        spc_version = self.get_attribute('spc_version')
        package_suffix = "debug" if self.package_suffix == "debug" else None

        region1 = "-".join(filter(None, [chip_name, func_name]))
        region2 = ".".join(filter(None, [version, spc_version]))
        package_name = self.name_sep.join(filter(None, [
            region1, region2, os_arch, chip_plat, deploy_type, package_suffix, ext_name,
        ]))

        return f"{package_name}.{self.suffix}"


class MakeselfPkgParams(NamedTuple):
    """run包打包参数。"""
    makeself_tool: str
    makeself_header: str
    help_info: str
    source_target: str
    package_name: str
    comments: str
    install_script: str
    cleanup: Optional[str] = None


def get_func_name(func_name: str, package_attr) -> str:
    """获取包func_name。"""
    return func_name or package_attr.get('func_name')


def chmod_before_package(pkg_mods: List[PkgMod], release_dir: Union[str, Path]):
    """打包前修改包内文件权限。"""
    # pkg_mods按路径反序
    for pkg_mod in sorted(pkg_mods, key=attrgetter('path'), reverse=True):
        filepath = os.path.join(release_dir, pkg_mod.path)
        os.chmod(filepath, pkg_mod.mod)


def softlink_before_package(pkg_softlinks: List[PkgSoftlink], release_dir: Union[str, Path]):
    """打包前创建包内文件软链。"""
    for pkg_softlink in pkg_softlinks:
        src_path = os.path.join(release_dir, pkg_softlink.src_path)
        dst_path = os.path.join(release_dir, pkg_softlink.dst_path)
        os.symlink(
            os.path.relpath(src_path, os.path.dirname(dst_path)), dst_path
        )


def compose_makeself_command(params: MakeselfPkgParams) -> str:
    """组装makeself包打包命令。"""

    def get_cleanup_commands() -> List[str]:
        if params.cleanup:
            return ['--cleanup', params.cleanup]
        return []
    
    commands = chain(
        [
            '--pigz', '--complevel', '4',
            '--nomd5', '--sha256', '--nooverwrite', '--chown', '--tar-format', 'gnu',
            '--tar-extra', '--numeric-owner', '--tar-quietly'
        ],
        get_cleanup_commands(),
        ['./', params.package_name, params.comments]
    )
    
    command = ' '.join(commands)
    return command


def create_makeself_pkg_params_factory(source_target: str,
                                       package_name: str,
                                       comments: str
                                       ) -> Callable[[str, str, Dict], MakeselfPkgParams]:
    """创建Makeself打包参数工厂。"""

    def create_makeself_pkg_params(top_dir: str,
                                   delivery_dir: str,
                                   package_attr: Dict) -> MakeselfPkgParams:
        """创建Makeself打包参数。"""

        def get_help_info_relpath():
            return os.path.join(source_target, str(package_attr.get('help')))

        install_script = str(package_attr.get('install_script'))
        help_info = get_help_info_relpath()
        cleanup = package_attr.get('cleanup')

        relative_path = os.path.relpath(
            os.path.join(top_dir, 'open_source', 'makeself'), delivery_dir
        )
        makeself_tool = os.path.join(relative_path, 'makeself.sh')
        makeself_header = os.path.join(relative_path, 'makeself-header.sh')

        params = MakeselfPkgParams(
            makeself_tool=makeself_tool,
            makeself_header=makeself_header,
            help_info=help_info,
            source_target=source_target,
            package_name=package_name,
            comments=comments,
            install_script=install_script,
            cleanup=cleanup,
        )

        return params

    return create_makeself_pkg_params


def create_run_package_command(delivery_dir: str,
                               params: MakeselfPkgParams
                               ) -> Tuple[Optional[str], Optional[str]]:
    """
    功能描述: 组装打run包命令
    返回值: command
    """
    source_target = params.source_target
    install_script = params.install_script
    help_info = params.help_info
    cleanup = params.cleanup
    makeself_tool = params.makeself_tool
    makeself_header = params.makeself_header
    print("ybh " + delivery_dir + source_target + install_script)
    print(os.path.join(delivery_dir, source_target, install_script))
    print(os.path.join(delivery_dir, help_info))
    #print(os.path.join(delivery_dir, source_target, cleanup))
    """
    if not os.path.isfile(os.path.join(delivery_dir, source_target, install_script)):
        return None, f"run_install_script({install_script}) doesn't exist."

    if not os.path.isfile(os.path.join(delivery_dir, help_info)):
        return None, f"run_help({help_info}) doesn't exist."

    if cleanup and not os.path.isfile(os.path.join(delivery_dir, source_target, cleanup)):
        return None, f"run_cleanup({cleanup}) doesn't exist."

    if not os.path.isfile(os.path.join(delivery_dir, makeself_tool)):
        return None, f"makeself_tool({makeself_tool}) doesn't exist."

    if not os.path.isfile(os.path.join(delivery_dir, makeself_header)):
        return None, f"makeself_header({makeself_header}) doesn't exist."
    """
    return compose_makeself_command(params), None


def is_strip_package_dir(package_name: PackageName, package_attr: Dict[str, str]) -> bool:
    """打包（tar包）时是否剥离第一级目录。"""
    if package_name.func_name in ["testcase"]:
        return True
    attr_value = package_attr.get('strip_package_dir', '').lower()
    if attr_value == 'true':
        return True
    return False


def create_testcase_package_command(package_name: str,
                                    source_target: str
                                    ) -> Tuple[Optional[str], Optional[str]]:
    """创建testcase包打包命令。"""
    cmd = " ".join([
        "cd", source_target, "&&", "tar", "-zcf", os.path.join("..", package_name), "*"
    ])
    return cmd, None


def get_tar_extra_options(package_attr: Dict[str, str]) -> Iterator[str]:
    """获取tar命令额外参数。"""
    pkg_own = package_attr.get('pkg_own')
    if pkg_own:
        parts = pkg_own.split(':')
        owner = parts[0]
        group = parts[1]
        yield from [f'--owner={owner}', f'--group={group}']

def add_rpm_dir_attr(item, spec_param):
    """
    功能描述: 生成rpm目录权限命令
    返回值: command
    """
    install_path = item['value']

    # 如果用户和属组传入的是变量，则设置为spec中的变量，否则传入具体值
    if item['install_own'] == '$username:$usergroup':
        username = '%{username}'
        usergroup = '%{usergroup}'
    else:
        own_parts = item['install_own'].split(":")
        username = own_parts[0]
        usergroup = own_parts[1]
    attr_command = f'%attr ({item["install_mod"]}, {username}, {usergroup}) /{install_path}'
    
    spec_param[2].append(attr_command + '\n')
    spec_param[3].add(install_path)
    return spec_param


def create_spec_command(spec_file_path, delivery_dir, spec_param, pkg_name):
    """
    功能描述: 生成spec文件，组装rpm打包命令
    返回值: command
    """
    for install_path in spec_param[3]:
        spec_param[1].insert(0, 'mkdir -p %{buildroot}/' + install_path + '\n')

    spec_filename = pkg_name + '.spec'
    with open(spec_file_path + '/' + spec_filename, 'r') as file:
        lines = file.readlines()

    # 新增行字典，键为标签，值为要插入的新行列表
    for i, line in enumerate(lines):
        if line.strip() == "#SOURCE":
            # 在#SOURCE标签后插入新行
            lines[i + 1:i + 1] = spec_param[0]
    for i, line in enumerate(lines):
        if line.startswith("%install"):
            # 在%install标签后插入新行
            lines[i + 1:i + 1] = spec_param[1]
    for i, line in enumerate(lines):
        if line.startswith("%defattr"):
            # 在%defattr标签后插入新行
            lines[i + 1:i + 1] = spec_param[2]

    # 写回修改后的.spec文件
    dst_spec_filename = delivery_dir + '/' + spec_filename
    with open(dst_spec_filename, 'w') as file:
        file.writelines(lines)
    return dst_spec_filename

def get_header_files(directory):
    header_files = []
    for path in Path(directory).rglob('*.h'):
        # 路径转换为字符串形式，并去掉目录部分
        header_files.append(str(path.relative_to(directory)))
    return header_files

def gen_spec_command(item, spec_param):
    """
    功能描述: 根据打包xml文件生成对应的spec命令
    参数：spec_param列表，有5个参数
        0.Source文件列表
        1.install文件列表
        2.设置安装文件权限列表
        3.创建安装目录列表
        4.源文件编号
    返回值: command
    """
    install_path = item['install_path']

    # 如果用户和属组传入的是变量，则设置为spec中的变量，否则传入具体值
    if item['install_own'] == '$username:$usergroup':
        username = '%{username}'
        usergroup = '%{usergroup}'
    else:
        own_parts = item['install_own'].split(":")
        username = own_parts[0]
        usergroup = own_parts[1]

    attr_command = f'%attr ({item["install_mod"]}, {username}, {usergroup}) /{install_path}/{item["value"]}'
    install_command = '} %{buildroot}/' + f'{install_path}'

    if item['value'][-1] == '/':
        header_files = get_header_files(
            os.path.join(str(Path(__file__).resolve().parents[4]),
                         item['src_path'], item['value']))
        for header_file in header_files:
            file_num = str(spec_param[4])
            spec_param[0].append(f'Source{file_num}:        {header_file}\n')
            spec_param[2].append(attr_command + f'/{header_file}\n')
            spec_param[1].append('cp -a %{SOURCE' + file_num + install_command + f'/{item["value"]}\n')
            spec_param[4] += 1
        spec_param[3].add(install_path + item['value'])
        item['value'] += '*.h'
    else:
        file_num = str(spec_param[4])
        spec_param[0].append(f'Source{file_num}:        {item["value"]}\n')
        spec_param[2].append(attr_command + '\n')
        spec_param[1].append('cp -a %{SOURCE' + file_num + install_command + '\n')
        spec_param[3].add(install_path)
        spec_param[4] += 1
    return spec_param

def create_rpm_package_command(package_name: PackageName, spec_file_name) -> Iterator[str]:
    """
    功能描述: 组装rpm打包命令
    返回值: command
    """
    version_info = package_name.version
    os_arch = package_name.os_arch.replace(".aarch64","")
    spec_file = spec_file_name + '.spec'

    rpmbuild_cmd = f'cp {spec_file} $HOME/rpmbuild/ && ' +\
                    f'rpmbuild -ba --target=aarch64 --define "version {version_info}"' +\
                    f' --define "release {os_arch}" {spec_file}' +\
                    f' && cp $HOME/rpmbuild/RPMS/aarch64/*.rpm  {package_name.getvalue()}'
    return rpmbuild_cmd

def get_filter_extra_options(package_name: str) -> Iterator[str]:
    """获取压缩命令额外参数。"""
    if package_name.find('aicpu_syskernels') > 0:
        yield '-9'


def stripped_tar_package_commands(package_name: str,
                                  source_target: str,
                                  filter_cmd: str,
                                  tar_extra_options: Iterator[str],
                                  filter_extra_options: Iterator[str]) -> Iterator[str]:
    """移除顶层目录的txz包打包命令。"""
    return chain(
        ('cd', source_target, '&&', 'tar', '--numeric-owner', '-c'), tar_extra_options,
        ('-f-', '*', '|', filter_cmd), filter_extra_options,
        ('>', os.path.join("..", package_name))
    )


def normal_tar_package_commands(package_name: str,
                                source_target: str,
                                filter_cmd: str,
                                tar_extra_options: Iterator[str],
                                filter_extra_options: Iterator[str]) -> Iterator[str]:
    """普通txz包打包命令。"""
    return chain(
        ('tar', '--numeric-owner', '-c'), tar_extra_options,
        ('-f-', source_target, '|', filter_cmd), filter_extra_options,
        ('>', package_name)
    )


def tar_package_commands(package_name: PackageName,
                         source_target: str,
                         package_attr: Dict[str, str],
                         filter_cmd: str) -> Iterator[str]:
    """创建tar包打包命令。"""
    output_name = package_name.getvalue()
    tar_extra_options = get_tar_extra_options(package_attr)
    filter_extra_options = get_filter_extra_options(output_name)

    if is_strip_package_dir(package_name, package_attr):
        return stripped_tar_package_commands(
            output_name, source_target, filter_cmd, tar_extra_options, filter_extra_options)

    return normal_tar_package_commands(
        output_name, source_target, filter_cmd, tar_extra_options, filter_extra_options)


def create_rar_package_command(package_name: str,
                               source_target: str
                               ) -> Tuple[Optional[str], Optional[str]]:
    """创建rar打包命令。"""
    cmd = " ".join(["zip -ry1q", package_name, source_target])
    return cmd, None


def pkg_cmds_to_str(cmds: Iterator[str]) -> str:
    """打包命令转为字符串。"""
    return ' '.join(cmds)


def print_package_info(print_func: Callable[..., None], filter_func, output: str):
    """打印打包信息。"""
    invoke(
        pipe(
            strip_lines,
            filter_func,
            partial(map, methodcaller('replace', '%', '%%')),
            partial(map, print_func),
            list,
        ),
        output.splitlines()
    )


def check_limit_size(filepath: str, limit_size: int) -> Tuple[bool, int]:
    """检查包大小是否超限。"""
    file_size = os.path.getsize(filepath)
    if file_size > limit_size:
        return False, file_size
    return True, file_size


def exec_pack_cmd(delivery_dir: str,
                  pack_cmd: str,
                  package_name: str,
                  quiet: int = 1) -> str:
    """执行打包命令。"""
    if delivery_dir:
        cmd = f'cd {delivery_dir} && {pack_cmd}'
    else:
        cmd = pack_cmd

    COMM_LOG.cilog_info(THIS_FILE_NAME, "pacakge cmd: %s", cmd)
    result = subprocess.run(cmd, shell=True, check=False, stdout=PIPE, stderr=STDOUT)
    output = result.stdout.decode()
    if result.returncode != 0:
        COMM_LOG.cilog_error(
            THIS_FILE_NAME, "compress package(%s) failed! %s.",
            package_name, output
        )
        raise CompressError(package_name)
    print_func = partial(COMM_LOG.cilog_info, THIS_FILE_NAME)
    if quiet:
        print_package_info(
            print_func,
            partial(filter, methodcaller('startswith', 'About to compress')),
            output,
        )
    else:
        print_package_info(
            print_func,
            identity,
            output,
        )

    return package_name


def get_spec_property(spec_path: str, property_: str) -> str:
    """获取spec文件Name属性。"""
    for line in iter_text(spec_path):
        if line.startswith(f'{property_}:'):
            return line.strip().split()[1]
    raise GetSpecPropertyError(property_)


def pack_sph_rpm(delivery_dir: str, spec_path: str) -> str:
    """打包补丁rpm包。"""
    name = get_spec_property(spec_path, 'Name')
    version = get_spec_property(spec_path, 'Version')
    release = get_spec_property(spec_path, 'Release')
    release_real = subprocess.run(
        ['rpm', '--eval', release], check=False, stdout=PIPE, stderr=STDOUT
    ).stdout.decode().strip()
    arch = 'aarch64'
    package_name = '.'.join(['-'.join([name, version, release_real]), arch, 'rpm'])

    rpmbuild_path = os.path.join(os.path.expanduser('~'), 'rpmbuild')

    makedirs = pipe(
        partial(map, partial(os.path.join, rpmbuild_path)),
        partial(map, partial(os.makedirs, exist_ok=True)),
        tuple,
    )
    makedirs(['BUILD', 'BUILDROOT', 'RPMS', 'SOURCES', 'SPECS', 'SRPMS'])

    pack_cmd = f'rpmbuild -bb --target=aarch64 {spec_path}'
    exec_pack_cmd(None, pack_cmd, package_name)

    delivery_path = os.path.join(delivery_dir, package_name)
    if os.path.exists(delivery_path):
        os.remove(delivery_path)
    shutil.move(
        os.path.join(rpmbuild_path, 'RPMS', arch, package_name), delivery_path
    )

    return package_name
