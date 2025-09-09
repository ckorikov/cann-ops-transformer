#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# error number and description
OPERATE_FAILED="0x0001"
PARAM_INVALID="0x0002"
FILE_NOT_EXIST="0x0080"
FILE_NOT_EXIST_DES="File not found."
FILE_WRITE_FAILED="0x0081"
FILE_WRITE_FAILED_DES="File write failed."
FILE_READ_FAILED="0x0082"
FILE_READ_FAILED_DES="File read failed."
FILE_REMOVE_FAILED="0x0090"
FILE_REMOVE_FAILED_DES="Failed to remove file."
UNAME_NOT_EXIST="0x0091"
UNAME_NOT_EXIST_DES="Username not exists."
OPP_COMPATIBILITY_CEHCK_ERR="0x0092"
OPP_COMPATIBILITY_CEHCK_ERR_DES="OppTransformer compatibility check error."
PERM_DENIED="0x0093"
PERM_DENIED_DES="Permission denied."

OPP_PLATFORM_DIR=ops_transformer
OPP_PLATFORM_UPPER=$(echo "${OPP_PLATFORM_DIR}" | tr '[:lower:]' '[:upper:]')
CURR_OPERATE_USER="$(id -nu 2>/dev/null)"
CURR_OPERATE_GROUP="$(id -ng 2>/dev/null)"
# defaults for general user
if [ "$(id -u)" != "0" ]; then
  DEFAULT_USERNAME="${CURR_OPERATE_USER}"
  DEFAULT_USERGROUP="${CURR_OPERATE_GROUP}"
  DEFAULT_INSTALL_PATH="${HOME}/Ascend"
else
  IS_FOR_ALL="y"
  DEFAULT_USERNAME="root"
  DEFAULT_USERGROUP="root"
  DEFAULT_INSTALL_PATH="/usr/local/Ascend"
fi

# run package's files info, CURR_PATH means current temp path
CURR_PATH=$(dirname $(readlink -f $0))
FILELIST_FILE="${CURR_PATH}/../../filelist.csv"
INSTALL_SHELL_FILE="${CURR_PATH}/opp_install.sh"
UPGRADE_SHELL_FILE="${CURR_PATH}/opp_upgrade.sh"
RUN_PKG_INFO_FILE="${CURR_PATH}/../scene.info"
VERSION_INFO_FILE="${CURR_PATH}/../../version.info"
COMMON_INC_FILE="${CURR_PATH}/common_func.inc"
VERCHECK_FILE="${CURR_PATH}/ver_check.sh"
PRE_CHECK_FILE="${CURR_PATH}/../bin/prereq_check.bash"
VERSION_COMPAT_FUNC_PATH="${CURR_PATH}/version_compatiable.inc"
COMMON_FUNC_V2_PATH="${CURR_PATH}/common_func_v2.inc"
VERSION_CFG_PATH="${CURR_PATH}/version_cfg.inc"
OPP_COMMON_FILE="${CURR_PATH}/opp_common.sh"

. "${VERSION_COMPAT_FUNC_PATH}"
. "${COMMON_INC_FILE}"
. "${COMMON_FUNC_V2_PATH}"
. "${VERSION_CFG_PATH}"
. "${OPP_COMMON_FILE}"

ARCH_INFO=$(grep -e "arch" "$RUN_PKG_INFO_FILE" | cut --only-delimited -d"=" -f2-)

# defaluts info determinated by user's inputs
ASCNED_INSTALL_INFO="ascend_install.info"
TARGET_INSTALL_PATH="${DEFAULT_INSTALL_PATH}" #--input-path
TARGET_USERNAME="${CURR_OPERATE_USER}"
TARGET_USERGROUP="${CURR_OPERATE_GROUP}"
TARGET_MOULDE_DIR="" # TARGET_INSTALL_PATH + PKG_VERSION_DIR + OPP_PLATFORM_DIR
TARGET_DIR=""        # TARGET_INSTALL_PATH + PKG_VERSION_DIR

# keys of infos in ascend_install.info
KEY_INSTALLED_UNAME="USERNAME"
KEY_INSTALLED_UGROUP="USERGROUP"
KEY_INSTALLED_TYPE="${OPP_PLATFORM_UPPER}_INSTALL_TYPE"
KEY_INSTALLED_PATH="${OPP_PLATFORM_UPPER}_INSTALL_PATH_VAL"
KEY_INSTALLED_VERSION="${OPP_PLATFORM_UPPER}_VERSION"
KEY_INSTALLED_FEATURE="${OPP_PLATFORM_UPPER}_INSTALL_FEATURE"
KEY_INSTALLED_CHIP="${OPP_PLATFORM_UPPER}_INSTALL_CHIP"

# keys of infos in run package
KEY_RUNPKG_VERSION="Version"

# init install cmd status, set default as n
CMD_LIST="$*"
IS_UNINSTALL=n
IS_INSTALL=n
IS_UPGRADE=n
IS_QUIET=n
IS_INPUT_PATH=n
IS_CHECK=n
IS_PRE_CHECK=n
IN_INSTALL_TYPE=""
IN_INSTALL_PATH=""
IS_DOCKER_INSTALL=n
IS_SETENV=n
DOCKER_ROOT=""
CONFLICT_CMD_NUMS=0
IN_FEATURE="All"

# log functions
# start info before shell executing
startlog() {
  echo "[OpsTransformer] [$(getdate)] [INFO]: Start Time: $(getdate)"
}

exitlog() {
  echo "[OpsTransformer] [$(getdate)] [INFO]: End Time: $(getdate)"
}

check_group_with_user() {
  local ugroup="$1"
  local uname_param="$2"
  if [ $(groups "${uname_param}" | grep "${uname_param} :" -c) -eq 1 ]; then
    related=$(groups "${uname_param}" 2>/dev/null | awk -F":" '{print $2}' | grep -w "${ugroup}")
  else
    related=$(groups "${uname_param}" 2>/dev/null | grep -w "${ugroup}")
  fi
  if [ "${related}" != "" ]; then
    return 0
  else
    logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Input empty username or usergroup is invalid."
    return 1
  fi
}

#check ascend_install.info for the change in code warning
get_installed_info() {
  local key="$1"
  local res=""
  if [ -f "${INSTALL_INFO_FILE}" ]; then
    chmod 644 "${INSTALL_INFO_FILE}" >/dev/null 2>&1
    res=$(cat ${INSTALL_INFO_FILE} | grep "${key}" | awk -F = '{print $2}')
  fi
  echo "${res}"
}

precleanbeforeinstall() {
  _installed_path=$(get_installed_info "${KEY_INSTALLED_PATH}")
  _files_existed=1
  # check the installation folder has files or opp module existed or not
  _existed_files=$(find ${TARGET_MOULDE_DIR} -path ${TARGET_MOULDE_DIR}/aicpu -prune -o -type f -print 2>/dev/null)
  _arrs="aicpu Ascend310 Ascend910 Ascend310P Ascend"

  for i in ${_arrs}; do
    _existed_files=$(find ${TARGET_MOULDE_DIR} -path ${TARGET_MOULDE_DIR}/${i} -prune -o -type f -print 2>/dev/null)
    if [ "${_existed_files}" != "" ]; then
      break
    fi
  done

  if [ "${_existed_files}" = "" ]; then
    _files_existed=1
  else
    _files_existed=0
    logandprint "[WARNING]: Install folder has files existed. Some files are listed below:"
    _ret_array=$(echo ${_existed_files} | awk '{split($0,arr," ");for(i in arr) print arr[i]}')
    for i in $_ret_array; do
      id=$((${id:=-1} + 1))
      eval _ret_array_$id=$i
    done
    for idx in $(seq 0 5); do
      if [ "$(eval echo '$'_ret_array_$idx)" != "" ]; then
        logandprint "[WARNING]: file: "$(eval echo '$'_ret_array_$idx)""
      fi
    done
  fi

  if [ "${_files_existed}" = "0" ]; then
    if [ "${IS_QUIET}" = y ]; then
      logandprint "[WARNING]: Directory has file existed or installed opp\
 module, are you sure to keep installing opp module in it? y"
    else
      if [ ! -f "${TARGET_MOULDE_DIR}/ascend_install.info" ]; then
        logandprint "[INFO]: Directory has file existed, do you want to continue? [y/n]"
      else
        logandprint "[INFO]: Opp package has been installed on the path $(get_installed_info "${KEY_INSTALLED_PATH}"),\
 the version is $(get_installed_info "${KEY_INSTALLED_VERSION}"),\
 and the version of this package is ${RUN_PKG_VERSION}, do you want to continue? [y/n]"
      fi
      while true; do
        read yn
        if [ "$yn" = n ]; then
          logandprint "[INFO]: Exit to install opp module."
          exitlog
          exit 0
        elif [ "$yn" = y ]; then
          break
        else
          echo "[WARNING]: Input error, please input y or n to choose!"
        fi
      done
    fi
  else
    logandprint "[INFO]: Directory is empty, directly install opp module."
  fi

  ret=0
  if [ $(id -u) -eq 0 ]; then
    parent_dirs_permission_check "${_path}" && ret=$? || ret=$?
    if [ "${IS_QUIET}" = "y" ]; then
      if [ ${ret} -ne 0 ]; then
        logandprint "[ERROR]: ERR_NO:0x0095;ERR_DES:the given dir, or its parents, permission is invalid.\
 Please install without quiet mode and check permission."
        exitlog
        exit 1
      fi
    else
      if [ ${ret} -ne 0 ]; then
        logandprint "[WARNING]: You are going to put run-files on a unsecure install-path, do you want to continue? [y/n]"
        while true; do
          read yn
          if [ "$yn" = n ]; then
            exit 1
          elif [ "$yn" = y ]; then
            break
          else
            echo "[ERROR]: ERR_NO:0x0002;ERR_DES:input error, please input again!"
          fi
        done
      fi
    fi
  fi

  if [ "${_files_existed}" = "0" ] && [ "${_installed_path}" = "${_path}" ]; then
    logandprint "[INFO]: Clean the installed opp module before install."
    if [ ! -f "${UNINSTALL_SHELL_FILE}" ]; then
      logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
 (${UNINSTALL_SHELL_FILE}) not exists. Please set the correct install \
 path or clean the previous version opp install info (/etc/ascend_install.info) and then reinstall it."
      return 1
    fi
    uninstall_file_path="${TARGET_MOULDE_DIR}/script"
    keyword=$(cat "${uninstall_file_path}/uninstall.sh" | grep function)
    if [ "${keyword}" != "" ]; then
      #uninstall the file before compatible
      bash "${uninstall_file_path}/uninstall.sh"
      if [ "$?" != 0 ]; then
        logandprint "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Clean the installed directory failed."
        return 1
      fi
    else
      bash "${UNINSTALL_SHELL_FILE}" "${TARGET_DIR}" "uninstall" "${IS_QUIET}" $IN_FEATURE "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}"
      if [ "$?" != 0 ]; then
        logandprint "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Clean the installed directory failed."
        return 1
      fi
    fi

  fi

}

parent_dirs_permission_check() {
  current_dir="$1" parent_dir="" short_install_dir=""
  owner="" mod_num=""

  parent_dir=$(dirname "${current_dir}")
  short_install_dir=$(basename "${current_dir}")
  logandprint "[INFO]: parent_dir value is [${parent_dir}] and children_dir value is [${short_install_dir}]"

  if [ "${current_dir}"x = "/"x ]; then
    logandprint "[INFO]: parent dirs permission checked successfully"
    return 0
  else
    owner=""
    if [ -d "${parent_dir}"/"${short_install_dir}" ]; then
      owner=$(stat -c %U "${parent_dir}"/"${short_install_dir}")
    fi
    if [ "${owner}" = "" ]; then
      logandprint "[WARNING]: [${parent_dir}/${short_install_dir}] is not exist."
      return 1
    elif [ "${owner}" != "root" ]; then
      logandprint "[WARNING]: [${short_install_dir}] permission isn't right, it should belong to root."
      return 1
    fi
    logandprint "[INFO]: [${short_install_dir}] belongs to root."

    mod_num=$(stat -c %a "${parent_dir}"/"${short_install_dir}")
    if [ ${mod_num} -lt 755 ]; then
      logandprint "[WARNING]: [${short_install_dir}] permission is too small, it is recommended that the permission be 755 for the root user."
      return 2
    elif [ ${mod_num} -eq 755 ]; then
      logandprint "[INFO]: [${short_install_dir}] permission is ok."
    else
      logandprint "[WARNING]: [${short_install_dir}] permission is too high, it is recommended that the permission be 755 for the root user."
      [ "${IS_QUIET}" = n ] && return 3
    fi

    parent_dirs_permission_check "${parent_dir}"
  fi
}

getfirstnotexistdir() {
  in_tmp_dir="$1"
  arr=$(echo ${in_tmp_dir} | tr '/' "\n")
  index=0
  id=-1
  for i in $arr; do
    id=$((${id:=-1} + 1))
    eval arr_$id=$i
    index=$(expr $index + 1)
  done
  len=${index}
  tmp_dir_0=""
  tmp_str_val=""
  for i in $(seq 0 ${len}); do
    tmp_str_val="${tmp_str_val}""/""$(eval echo '$'arr_$i)"
    eval tmp_dir_$i=${tmp_str_val}
  done
  for i in $(seq 0 ${len}); do
    if [ ! -d "$(eval echo '$'tmp_dir_$i)" ]; then
      echo "$(eval echo '$'tmp_dir_$i)"
      break
    fi
  done
}

checkprefolderspermission() {
  in_tmp_dir_value="$1"
  _uname_info="$2"
  arr_val=$(echo ${in_tmp_dir_value} | awk '{split($0,arr,"/");for(i in arr) print arr[i]}')
  index_value=0
  for i in $arr_val; do
    id=$((${id:=-1} + 1))
    eval arr_$id=$i
    index_value=$(expr $index_value + 1)
  done
  len_val="${index_value}"
  tmp_str=""
  for i in $(seq 0 ${len_val}); do
    tmp_str_info="${tmp_str}""/""$(eval echo '$'arr_$i)"
    eval tmp_dir_$i=${tmp_str_info}
  done
  for i in $(seq 0 ${len_val}); do
    if [ -d "$(eval echo '$'tmp_dir_$i)" ]; then
      # general user only can install for himself
      cd $(eval echo '$'tmp_dir_$i) >>/dev/null 2>&1
      if [ "$?" != "0" ]; then
        logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:The ${_uname_info} do\
 not have the permission to access $(eval echo '$'tmp_dir_$i), please reset the directory to a right permission."
        return 1
      fi
      cd - >>/dev/null 2>&1
    fi
  done
  return 0
}

select_last_dir_component() {
  path="$1"
  last_component=$(basename ${path})
  if [ "${last_component}" = "atc" ]; then
    last_component="atc"
    return
  elif [ "${last_component}" = "fwkacllib" ]; then
    last_component="fwkacllib"
    return
  elif [ "${last_component}" = "compiler" ]; then
    last_component="compiler"
    return
  fi
}

check_version_file() {
  pkg_path="$1"
  component_ret="$2"
  run_pkg_path_temp=$(dirname "${pkg_path}")
  run_pkg_path_temp2=${run_pkg_path_temp%/*}
  run_pkg_path="${run_pkg_path_temp}""/${component_ret}"
  run_pkg_path_temp2=${run_pkg_path%/*}
  version_file="${run_pkg_path}""/version.info"
  version_file_tmp="${run_pkg_path_temp2}""/version.info"
  if [ -f "${version_file_tmp}" ]; then
    version_file=${version_file_tmp}
  fi
  if [ -f "${version_file}" ]; then
    echo "${version_file}" 2 >>/dev/null
  else
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The [${component_ret}] version.info in path [${pkg_path}] not exists."
    exitlog
    exit 1
  fi
  return
}

check_opp_version_file() {
  if [ -f "${CURR_PATH}/../../version.info" ]; then
    opp_ver_info="${CURR_PATH}/../../version.info"
  elif [ -f "${DEFAULT_INSTALL_PATH}/${OPP_PLATFORM_DIR}/version.info" ]; then
    opp_ver_info="${DEFAULT_INSTALL_PATH}/${OPP_PLATFORM_DIR}/version.info"
  else
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The [${OPP_PLATFORM_DIR}] version.info not exists."
    exitlog
    exit 1
  fi
  return
}

check_relation() {
  opp_ver_info_val="$1"
  req_pkg_name="$2"
  req_pkg_version="$3"
  if [ -f "${COMMON_INC_FILE}" ]; then
    . "${COMMON_INC_FILE}"
    check_pkg_ver_deps "${opp_ver_info_val}" "${req_pkg_name}" "${req_pkg_version}"
    ret_situation=$ver_check_status
  else
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The ${COMMON_INC_FILE} not exists."
    exitlog
    exit 1
  fi
  return
}

show_relation() {
  relation_situation="$1"
  req_pkg_name_val="$2"
  req_pkg_path="$3"
  if [ "$relation_situation" = "SUCC" ]; then
    logandprint "[INFO]: Relationship of opp with ${req_pkg_name_val} in path ${req_pkg_path} checked successfully"
  else
    logandprint "[WARNING]: Relationship of opp with ${req_pkg_name_val} in path ${req_pkg_path} checked failed."
  fi
  return
}

find_version_check() {
  if [ "$(id -u)" != "0" ]; then
    atc_res=$(find ${HOME} -name "ccec_compiler" | grep Ascend | grep atc)
    fwk_res=$(find ${HOME} -name "ccec_compiler" | grep Ascend | grep fwk)
    comp_res=$(find ${HOME} -name "ccec_compiler" | grep Ascend | grep Ascend/compiler)
    ccec_compiler_path="$atc_res $fwk_res $comp_res"
  else
    atc_res=$(find /usr/local -name "ccec_compiler" | grep Ascend | grep atc)
    fwk_res=$(find /usr/local -name "ccec_compiler" | grep Ascend | grep fwk)
    comp_res=$(find /usr/local -name "ccec_compiler" | grep Ascend | grep Ascend/compiler)
    ccec_compiler_path="$atc_res $fwk_res $comp_res"
  fi
  check_opp_version_file
  ret_check_opp_version_file=$opp_ver_info
  for var in ${ccec_compiler_path}; do
    run_pkg_path_val=$(dirname "${var}")
    # find run pkg name
    select_last_dir_component "${run_pkg_path_val}"
    ret_pkg_name=$last_component
    #get check version
    check_version_file "${run_pkg_path_val}" "${ret_pkg_name}"
    ret_check_version_file=$version_file
    #check relation
    check_relation "${ret_check_opp_version_file}" "${ret_pkg_name}" "${ret_check_version_file}"
    ret_check_relation_val=$ret_situation
    #show relation
    show_relation "${ret_check_relation_val}" "${ret_pkg_name}" "${run_pkg_path_val}"
  done
  return
}

path_version_check() {
  path_env_list="$1"
  check_opp_version_file
  ret_check_opp_version_file_name=$opp_ver_info
  path_list=$(echo "${path_env_list}" | cut -d"=" -f2)
  array=$(echo ${path_list} | awk '{split($0,arr,":");for(i in arr) print arr[i]}')
  for var in ${array}; do
    path_ccec_compile=$(echo ${var} | grep -w "ccec_compiler")
    if [ "${path_ccec_compile}" != "" ]; then
      pkg_path_val=$(dirname $(dirname "${path_ccec_compile}"))
      # find run pkg name
      select_last_dir_component "${pkg_path_val}"
      ret_pkg_name_val=$last_component
      #get check version
      check_version_file "${pkg_path_val}" "${ret_pkg_name_val}"
      ret_check_version_file_val=$version_file
      #check relation
      check_relation "${ret_check_opp_version_file_name}" "${ret_pkg_name}" "${ret_check_version_file_val}"
      ret_check_relation=$ret_situation
      #show relation
      show_relation "${ret_check_relation}" "${ret_pkg_name}" "${pkg_path_val}"
    else
      echo "the var_case does not contains ccec_compiler" 2 >>/dev/null
    fi
  done
  return
}

check_docker_path() {
  docker_path="$1"
  if [ "${docker_path}" != "/"* ]; then
    echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --docker-root\
 must with absolute path that which is start with root directory /. Such as --docker-root=/${docker_path}"
    exitlog
    exit 1
  fi
  if [ ! -d "${docker_path}" ]; then
    echo "[OpsTransformer] [ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${docker_path} not exist, please create this directory."
    exitlog
    exit 1
  fi
}

judgment_path() {
  . "${COMMON_INC_FILE}"
  check_install_path_valid "${1}"
  if [ $? -ne 0 ]; then
    echo "[OpsTransformer][ERROR]: The opp install path ${1} is invalid, only characters in [a-z,A-Z,0-9,-,_] are supported!"
    exitlog
    exit 1
  fi
}

check_install_path() {
  TARGET_INSTALL_PATH="$1"
  # empty patch check
  if [ "x${TARGET_INSTALL_PATH}" = "x" ]; then
    echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --install-path\
 not support that the install path is empty."
    exitlog
    exit 1
  fi
  # space check
  if echo "x${TARGET_INSTALL_PATH}" | grep -q " "; then
    echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --install-path\
 not support that the install path contains space character."
    exitlog
    exit 1
  fi
  # delete last "/"
  local temp_path="${TARGET_INSTALL_PATH}"
  temp_path=$(echo "${temp_path%/}")
  if [ x"${temp_path}" = "x" ]; then
    temp_path="/"
  fi
  # covert relative path to absolute path
  local prefix=$(echo "${temp_path}" | cut -d"/" -f1 | cut -d"~" -f1)
  if [ "x${prefix}" = "x" ]; then
    TARGET_INSTALL_PATH="${temp_path}"
  else
    prefix=$(echo "${RUN_PATH}" | cut -d"/" -f1 | cut -d"~" -f1)
    if [ x"${prefix}" = "x" ]; then
      TARGET_INSTALL_PATH="${RUN_PATH}/${temp_path}"
    else
      echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES: Run package path is invalid: $RUN_PATH"
      exitlog
      exit 1
    fi
  fi
  # covert '~' to home path
  local home=$(echo "${TARGET_INSTALL_PATH}" | cut -d"~" -f1)
  if [ "x${home}" = "x" ]; then
    local temp_path_value=$(echo "${TARGET_INSTALL_PATH}" | cut -d"~" -f2)
    if [ "$(id -u)" -eq 0 ]; then
      TARGET_INSTALL_PATH="/root$temp_path_value"
    else
      local home_path=$(eval echo "${USER}")
      home_path=$(echo "${home_path}%/")
      TARGET_INSTALL_PATH="$home_path$temp_path_value"
    fi
  fi
}

# execute prereq_check file
exec_pre_check() {
  bash "${PRE_CHECK_FILE}"
}

# execute prereq_check file and interact with user
interact_pre_check() {
  exec_pre_check
  if [ "$?" != 0 ]; then
    if [ "${IS_QUIET}" = y ]; then
      logandprint "[WARNING]: Precheck of opp module execute failed! do you want to continue install? y"
    else
      logandprint "[WARNING]: Precheck of opp module execute failed! do you want to continue install?  [y/n] "
      while true; do
        read yn
        if [ "$yn" = "n" ]; then
          echo "stop install opp module!"
          exit 1
        elif [ "$yn" = y ]; then
          break
        else
          echo "[WARNING]: Input error, please input y or n to choose!"
        fi
      done
    fi
  fi
}

check_dir_permission_for_common_user() {
  current_dir_val="$1"
  parent_dir_val=$(dirname "${current_dir_val}")
  if [ "${parent_dir_val}"x = "/"x ]; then
    return
  fi
  if [ -d "${current_dir_val}" ]; then
    mod_num_val=$(stat -c %a "${current_dir_val}")
    num_0=${mod_num_val:0:1}
    num_1=${mod_num_val:1:1}
    num_2=${mod_num_val:2:1}
    if [ ${num_0} -lt 7 ] || [ ${num_1} -lt 5 ] || [ ${num_2} -lt 5 ]; then
      logandprint "[ERROR]: ${current_dir_val} permission is ${mod_num_val}, this permission is too small, and it should be 755."
      exitlog
      exit 1
    elif [ ${num_0} -eq 7 ] && [ ${num_1} -eq 5 ] && [ ${num_2} -eq 5 ]; then
      logandprint "[INFO]: ${current_dir_val} permission is ok."
    else
      if [ "${IS_QUIET}" = y ]; then
        logandprint "[WARNING]: ${current_dir_val} permission is ${mod_num_val}, this permission is too high, do you want to continue install? y"
      else
        logandprint "[WARNING]: ${current_dir_val} permission is ${mod_num_val}, this permission is too high, do you want to continue install?  [y/n] "
        while true; do
          read yn
          if [ "$yn" = n ]; then
            logandprint "[INFO]: stop install opp module!"
            exitlog
            exit 1
          elif [ "$yn" = y ]; then
            break
          else
            logandprint "[WARNING]: Input error, please input y or n to choose!"
          fi
        done
      fi
    fi
  fi
  check_dir_permission_for_common_user "${parent_dir_val}"
}

#get the dir of xxx.run
#opp_install_path_curr=`echo "$2" | cut -d"/" -f2- `
# cut first two params from *.run
get_run_path() {
  RUN_PATH=$(echo "$2" | cut -d"-" -f3-)
  if [ x"${RUN_PATH}" = x"" ]; then
    RUN_PATH=$(pwd)
  else
    # delete last "/"
    RUN_PATH=$(echo "${RUN_PATH%/}")
    if [ "x${RUN_PATH}" = "x" ]; then
      # root path
      RUN_PATH=$(pwd)
    fi
  fi
}

check_arch() {
  local architecture=$(uname -m)
  # check platform
  if [ "${architecture}" != "${ARCH_INFO}" ]; then
    logandprint "[ERROR]: ERR_NO:${OPERATE_FAILED};ERR_DES:the architecture of the run package arch:${ARCH_INFO}\
  is inconsistent with that of the current environment ${architecture}. "
    exitlog
    exit 1
  fi
}

get_opts() {
  local i=0
  while true; do
    if [ "$1" = "" ]; then
      break
    fi
    if [ "$(expr substr "$1" 1 2)" = "--" ]; then
      ((i++))
    fi
    if [ $i -gt 2 ]; then
      break
    fi
    shift
  done

  if [ "$*" = "" ]; then
    echo "[ERROR]: ERR_NO:${PARAM_INVALID}; ERR_DES:Unrecognized parameters.Try './xxx.run --help for more information.'"
    exitlog
    exit 1
  fi

  while true; do
    # skip 2 parameters avoid run pkg and directory as input parameter
    case "$1" in
      --version)
        if [ -e "${VERSION_INFO_FILE}" ]; then
          . "${VERSION_INFO_FILE}"
          echo ${Version}
          exitlog
          exit 0
        else
          echo "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The version file (${VERSION_INFO_FILE}) not exists or without execute permission."
          exitlog
          exit 1
        fi
        ;;
      --run | --full | --devel)
        IN_INSTALL_TYPE=$(echo ${1} | awk -F"--" '{print $2}')
        IS_INSTALL=y
        ((CONFLICT_CMD_NUMS++))
        shift
        ;;
      --upgrade)
        IS_UPGRADE=y
        ((CONFLICT_CMD_NUMS++))
        shift
        ;;
      --uninstall)
        IS_UNINSTALL=y
        ((CONFLICT_CMD_NUMS++))
        shift
        ;;
      --install-path=*)
        IS_INPUT_PATH=y
        IN_INSTALL_PATH=$(echo ${1} | cut -d"=" -f2-)
        # check path
        judgment_path "${IN_INSTALL_PATH}"
        check_install_path "${IN_INSTALL_PATH}"
        shift
        ;;
      --quiet)
        IS_QUIET=y
        shift
        ;;
      --install-for-all)
        IS_FOR_ALL=y
        shift
        ;;
      --check)
        IS_CHECK=y
        ((CONFLICT_CMD_NUMS++))
        shift
        ;;
      --check-path=*)
        check_path=$1
        ((CONFLICT_CMD_NUMS++))
        shift
        ;;
      --pre-check)
        IS_PRE_CHECK=y
        shift
        ;;
      --setenv)
        IS_SETENV=y
        shift
        ;;
      --docker-root=*)
        IS_DOCKER_INSTALL=y
        DOCKER_ROOT=$(echo $1 | cut -d"=" -f2)
        check_docker_path ${DOCKER_ROOT}
        shift
        ;;
      -*)
        echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Unsupported parameters [$1],\
 operation execute failed. Please use [--help] to see the useage."
        exitlog
        exit 1
        ;;
      *)
        break
        ;;
    esac
  done
}

# pre-check
check_opts() {
  if [ "${CONFLICT_CMD_NUMS}" -eq 0 ] && [ "x${IS_PRE_CHECK}" = "xy" ]; then
    interact_pre_check
    exitlog
    exit 0
  fi

  if [ "${CONFLICT_CMD_NUMS}" != 1 ]; then
    echo "[OpsTransformer] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:\
 only support one type: full/run/devel/upgrade/uninstall/check, operation execute failed!\
 Please use [--help] to see the usage."
    exitlog
    exit 1
  fi
}

# init target_dir and log for install
init_env() {
  get_install_package_dir "TARGET_MOULDE_DIR" "${VERSION_INFO_FILE}" "${TARGET_INSTALL_PATH}" "${OPP_PLATFORM_DIR}"
  TARGET_DIR=$(dirname ${TARGET_MOULDE_DIR})
  # Splicing docker-root and install-path
  if [ "${IS_DOCKER_INSTALL}" = "y" ]; then
    # delete last "/"
    local temp_path_param="${DOCKER_ROOT}"
    local temp_path_val=$(echo "${temp_path_param%/}")
    if [ "x${temp_path_val}" = "x" ]; then
      temp_path_val="/"
    fi
    TARGET_DIR=${temp_path_val}${TARGET_DIR}
  fi

  UNINSTALL_SHELL_FILE="${TARGET_MOULDE_DIR}/script/opp_uninstall.sh"
  INSTALL_INFO_FILE="${TARGET_MOULDE_DIR}/${ASCNED_INSTALL_INFO}"
  is_multi_version_pkg "pkg_is_multi_version" "$VERSION_INFO_FILE"
  get_version_dir "PKG_VERSION_DIR" "$VERSION_INFO_FILE"
  get_package_version "RUN_PKG_VERSION" "$VERSION_INFO_FILE"

  # creat log folder and log file
  comm_init_log

  logandprint "[INFO]: Execute the opp run package."
  logandprint "[INFO]: OperationLogFile path: ${COMM_LOGFILE}."
  logandprint "[INFO]: Input params: $CMD_LIST"

  local installed_version=$(get_installed_info "${KEY_INSTALLED_VERSION}")
  if [ "${installed_version}" = "" ]; then
    logandprint "[INFO]: Version of installing opp module is ${RUN_PKG_VERSION}."
  else
    if [ "${RUN_PKG_VERSION}" != "" ]; then
      logandprint "[INFO]: Existed opp module version is ${installed_version},\
 the new opp module version is ${RUN_PKG_VERSION}."
    fi
  fi
}

check_pre_install() {
  if [ "${IS_CHECK}" = "y" ] && [ "${check_path}" = "" ]; then
    path_env_list_val=$(env | grep -w PATH)
    path_ccec_compile_val=$(echo ${path_env_list} | grep -w "ccec_compiler")
    if [ "${path_ccec_compile_val}" != "" ]; then
      path_version_check "${path_env_list_val}"
    else
      find_version_check
    fi
    exitlog
    exit 0
  fi

  if [ "${IS_CHECK}"="y" ] && [ "${check_path}" != "" ]; then
    VERCHECK_FILE="${CURR_PATH}""/ver_check.sh"
    if [ ! -f "${VERCHECK_FILE}" ]; then
      logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file (${VERCHECK_FILE}) not exists.\
 Please make sure that the opp module installed in (${VERCHECK_FILE}) and then set the correct install path."
    fi
    bash "${VERCHECK_FILE}" "${check_path}"
    exitlog
    exit 0
  fi

  local installed_user=$(get_installed_info "${KEY_INSTALLED_UNAME}")
  local installed_group=$(get_installed_info "${KEY_INSTALLED_UGROUP}")
  if [ "${installed_user}" != "" ] || [ "${installed_group}" != "" ]; then
    if [ "${installed_user}" != "${TARGET_USERNAME}" ] || [ "${installed_group}" != "${TARGET_USERGROUP}" ]; then
      logandprint "[ERROR]: ERR_NO:0x0095;ERR_DES:The user and group are not same with last installation,\
 do not support overwriting installation!"
      exitlog
      exit 1
    fi
  fi
}

#Support the installation script when the specified path (relative path and absolute path) does not exist
mkdir_install_path() {
  local base_dir=$(dirname ${TARGET_INSTALL_PATH})
  if [ ! -d ${base_dir} ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${base_dir} not exist, please create this directory."
    exitlog
    exit 1
  fi

  if [ -d "${TARGET_INSTALL_PATH}" ]; then
    test -w ${TARGET_INSTALL_PATH} >>/dev/null 2>&1
    if [ "$?" -ne 0 ]; then
      #All paths exist with write permission
      logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:The ${TARGET_USERNAME} do\
 access ${TARGET_INSTALL_PATH} failed, please reset the directory to a right permission."
      exit 1
    fi
  else
    test -w ${base_dir} >>/dev/null 2>&1
    if [ "$?" -ne 0 ]; then
      #All paths exist with write permission
      logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:The ${TARGET_USERNAME} do\
 access ${base_dir} failed, please reset the directory to a right permission."
      exit 1
    else
      comm_create_dir "${TARGET_INSTALL_PATH}" "750" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"
    fi
  fi
}

check_permission() {
  if [ "${IS_CHECK}" = "y" ]; then
    if [ -z "$PKG_VERSION_DIR" ]; then
      preinstall_check --install-path="${TARGET_DIR}" --docker-root="${DOCKER_ROOT}" --script-dir="${CURR_PATH}" --package="${OPP_PLATFORM_DIR}" --logfile="${COMM_LOGFILE}"
      if [ $? -ne 0 ]; then
        logandprint "[ERROR]: ERR_NO:${OPP_COMPATIBILITY_CEHCK_ERR};ERR_DES:Check the compatibility of opp package fail,please confirm the true version package."
        exit 1
      fi
    fi
  fi

  if [ "${IS_FOR_ALL}" = y ] && [ $(id -u) -ne 0 ]; then
    check_dir_permission_for_common_user "${TARGET_DIR}"
  fi
}

install_package() {
  if [ "${IS_INSTALL}" = "y" ]; then
    # precheck before install opp module
    if [ "${IS_PRE_CHECK}" = "y" ]; then
      interact_pre_check
    fi

    # devel mode no need set the correct install user
    if [ "${IN_INSTALL_TYPE}" != "devel" ]; then
      check_group_with_user "${TARGET_USERNAME}" "${TARGET_USERGROUP}"
      if [ "$?" != 0 ]; then
        logandprint "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Installation of opp module execute failed!"
        comm_log_operation "Install" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
      fi
    else
      check_group_with_user "${TARGET_USERGROUP}" "${TARGET_USERNAME}"
      devel_group_ret="$?"
      if [ "${devel_group_ret}" != "0" ]; then
        TARGET_USERNAME="${DEFAULT_USERNAME}"
        TARGET_USERGROUP="${DEFAULT_USERGROUP}"
        if [ "${IS_QUIET}" = "y" ]; then
          logandprint "[WARNING]: Input user name or user group is invalid.\
 Would you like to set as the default user name (${DEFAULT_USERNAME}),\
 user group (${DEFAULT_USERGROUP}) for devel mode? y"
        else
          logandprint "[WARNING]: Input user name or user group is invalid.\
 Would you like to set as the default user name (${DEFAULT_USERNAME}),\
 user group (${DEFAULT_USERGROUP}) for devel mode? [y/n]"
          while true; do
            read yn
            if [ "$yn" = "n" ]; then
              logandprint "[INFO]: Exit to install opp module."
              comm_log_operation "Install" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
            elif [ "$yn" = y ]; then
              break
            else
              echo "[WARNING]: Input error, please input y or n to choose!"
            fi
          done
        fi
      fi
    fi
    # install need check whether the custom user can access the folders or not, move to opp_install
    if [ $(id -u) -ne 0 ]; then
      checkprefolderspermission "${TARGET_MOULDE_DIR}" "${TARGET_USERNAME}"
    fi

    if [ "$?" != 0 ]; then
      comm_log_operation "Install" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
    fi

    # use uninstall to clean the install folder
    precleanbeforeinstall
    if [ "$?" != 0 ]; then
      comm_log_operation "Install" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
    fi

    _FIRST_NOT_EXIST_DIR=$(getfirstnotexistdir "${TARGET_MOULDE_DIR}")
    #getfirstnotexistdir "/usr/local/tmp/CANN-1.81/opp"
    is_the_last_dir_opp=""
    if [ "${_FIRST_NOT_EXIST_DIR}" = "${TARGET_MOULDE_DIR}" ]; then
      is_the_last_dir_opp=1
    else
      is_the_last_dir_opp=0
    fi
    # check the compatibility of opp
    if [ -z "$PKG_VERSION_DIR" ]; then
      preinstall_process --install-path="${TARGET_DIR}" --docker-root="${DOCKER_ROOT}" --script-dir="${CURR_PATH}" --package="${OPP_PLATFORM_DIR}" --logfile="${COMM_LOGFILE}"
      if [ $? -ne 0 ]; then
        logandprint "[ERROR]: ERR_NO:${OPP_COMPATIBILITY_CEHCK_ERR};ERR_DES:Check the compatibility of opp package fail,please confirm the true version package."
        exit 1
      fi
    fi
    bash "${INSTALL_SHELL_FILE}" "${TARGET_INSTALL_PATH}" "${DEFAULT_USERNAME}" "${DEFAULT_USERGROUP}" "${IN_FEATURE}" \
      "${IN_INSTALL_TYPE}" "${IS_QUIET}" "${_FIRST_NOT_EXIST_DIR}" "${is_the_last_dir_opp}" "${IS_FOR_ALL}" \
      "${IS_SETENV}" "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}" "${IS_INPUT_PATH}" "" ""
    if [ "$?" != 0 ]; then
      comm_log_operation "Install" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
    fi
    if [ $(id -u) -eq 0 ]; then
      chown -R "root":"root" "${TARGET_MOULDE_DIR}/script" 2>/dev/null
      chown "root":"root" "${TARGET_MOULDE_DIR}" 2>/dev/null
    else
      chmod -R 550 "${TARGET_MOULDE_DIR}/script" 2>/dev/null
      chmod 440 "${TARGET_MOULDE_DIR}/script/filelist.csv" 2>/dev/null
    fi
    comm_log_operation "Install" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
  fi
}

upgrade_package() {
  if [ "${IS_UPGRADE}" = "y" ]; then
    # if latest dir has no opp module exit first!
    if [ "$pkg_is_multi_version" = "true" ]; then
      get_package_upgrade_version_dir "upgrade_version_dir" "${TARGET_INSTALL_PATH}" "${OPP_PLATFORM_DIR}"
      if [ -z "${upgrade_version_dir}" ]; then
        logandprint "[ERROR]: latest path has no opp module"
        upgrade_path=$(ls "${TARGET_INSTALL_PATH}" 2>/dev/null)
        if [ "${upgrade_path}" = "" ]; then
          rm -rf "${TARGET_INSTALL_PATH}"
        fi
        exit 1
      fi
    fi
    if [ "${IS_FOR_ALL}" = y ] && [ $(id -u) -ne 0 ]; then
      check_dir_permission_for_common_user "${TARGET_DIR}"
    fi
    install_type=$(get_installed_info "${KEY_INSTALLED_TYPE}")
    # precheck before upgrade opp module
    if [ "${IS_PRE_CHECK}" = "y" ]; then
      interact_pre_check
    fi
    # check the compatibility of opp
    if [ -z "$PKG_VERSION_DIR" ]; then
      preinstall_process --install-path="${TARGET_DIR}" --docker-root="${DOCKER_ROOT}" --script-dir="${CURR_PATH}" --package="${OPP_PLATFORM_DIR}" --logfile="${COMM_LOGFILE}"
      if [ $? -ne 0 ]; then
        logandprint "[ERROR]: ERR_NO:${OPP_COMPATIBILITY_CEHCK_ERR};ERR_DES:Check the compatibility of opp package fail,please confirm the true version package."
        exit 1
      fi
    fi
    bash "${UPGRADE_SHELL_FILE}" "${TARGET_INSTALL_PATH}" "${DEFAULT_USERNAME}" "${DEFAULT_USERGROUP}" ${IN_FEATURE} \
      "${IS_QUIET}" "${IS_FOR_ALL}" "${IS_SETENV}" "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}" "${IS_INPUT_PATH}" "${IS_UPGRADE}" "" ""
    if [ $(id -u) -eq 0 ]; then
      chown -R "root":"root" "${TARGET_MOULDE_DIR}/script" 2>/dev/null
      chown "root":"root" "${TARGET_MOULDE_DIR}" 2>/dev/null
    fi
    if [ $(id -u) -eq 0 ]; then
      chmod -R 555 "${TARGET_MOULDE_DIR}/script" 2>/dev/null
      chmod 444 "${TARGET_MOULDE_DIR}/script/filelist.csv" 2>/dev/null
    else
      chmod -R 550 "${TARGET_MOULDE_DIR}/script" 2>/dev/null
      chmod 440 "${TARGET_MOULDE_DIR}/script/filelist.csv" 2>/dev/null
    fi
    # uprate precheck info to ${TARGET_DIR}/bin/prereq_check.bash
    logandprint "[INFO]: Set precheck info."
    comm_log_operation "Upgrade" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
  fi
}

uninstall_package() {
  if [ "${IS_UNINSTALL}" = "y" ]; then
    install_type=$(get_installed_info "${KEY_INSTALLED_TYPE}")

    if [ "$?" != 0 ]; then
      comm_log_operation "Uninstall" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
    fi
    if [ ! -f "${UNINSTALL_SHELL_FILE}" ]; then
      logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
 (${UNINSTALL_SHELL_FILE}) not exists. Please make sure that the opp module\
 installed in (${TARGET_DIR}) and then set the correct install path."
      uninstall_path=$(ls "${TARGET_INSTALL_PATH}" 2>/dev/null)
      if [ "${uninstall_path}" = "" ]; then
        rm -rf "${TARGET_INSTALL_PATH}"
      fi
      comm_log_operation "Uninstall" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
      exit 0
    fi
    bash "${UNINSTALL_SHELL_FILE}" "${TARGET_INSTALL_PATH}" "uninstall" "${IS_QUIET}" $IN_FEATURE "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}"
    # remove precheck info in ${TARGET_DIR}/bin/prereq_check.bash
    logandprint "[INFO]: Remove precheck info."

    comm_log_operation "Uninstall" "${IN_INSTALL_TYPE}" "OpsTransformer" "$?" "${CMD_LIST}"
  fi
}

pre_check_only() {
  if [ "${IS_PRE_CHECK}" = "y" ]; then
    exec_pre_check
    exit $?
  fi
}

main() {
  check_arch

  get_run_path "$@"

  startlog

  get_opts "$@"

  check_opts

  init_env

  check_pre_install

  mkdir_install_path

  check_permission

  install_package

  upgrade_package

  uninstall_package

  pre_check_only
}

main "$@"
exit 0
