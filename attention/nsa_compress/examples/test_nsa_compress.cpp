/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_matmul_v2.cpp
 * \brief
 */


#include "aclnnop/aclnn_nsa_compress.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

#include <iostream>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <map>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <filesystem>
#include <regex>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <iomanip>

namespace fs = std::filesystem;
using namespace std;

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)


#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

    return ACL_SUCCESS;
}

/**
 * @brief Read data from file
 * @param [in] filePath: file path
 * @param [out] fileSize: file size
 * @return read result
 */
bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0u) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

/**
 * @brief Write data to file
 * @param [in] filePath: file path
 * @param [in] buffer: data to write to file
 * @param [in] size: size to write
 * @return write result
 */
bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

void FreeResource(aclTensor *input, aclTensor *weight, aclIntArray *actSeqLenOptional, aclTensor *output,
                  void *inputDevAddr, void *weightDevAddr, void *outputDevAddr, uint64_t workspaceSize,
                  void *workspaceAddr, int32_t deviceId, aclrtStream *stream)
{
    // 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    if (input != nullptr) {
        aclDestroyTensor(input);
    }
    if (weight != nullptr) {
        aclDestroyTensor(weight);
    }
    if (actSeqLenOptional != nullptr) {
        aclDestroyIntArray(actSeqLenOptional);
    }
    if (output != nullptr) {
        aclDestroyTensor(output);
    }

    // 8. 释放device资源，需要根据具体API的接口定义修改
    if (inputDevAddr != nullptr) {
        aclrtFree(inputDevAddr);
    }
    if (weightDevAddr != nullptr) {
        aclrtFree(weightDevAddr);
    }
    if (outputDevAddr != nullptr) {
        aclrtFree(outputDevAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    // 释放device资源
    if (stream != nullptr) {
        aclrtDestroyStream(*stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main(int argc, char **argv)
{
    // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::string exeFile(argv[0]);
    std::string currentPath = std::string(exeFile.substr(0, exeFile.rfind('/')) + "/");

    ret = fs::exists(currentPath);
    CHECK_RET(ret == true,
              LOG_PRINT("nsa compress input path=%s not exists. please generate data first\n", currentPath.c_str());
              return ret);

    std::ifstream config_file(currentPath + "config.txt");
    std::map<std::string, int> configDict;
    std::string line;
    std::regex integer_regex("^[+-]?\\d+$"); // 匹配整数的正则表达式
    while (std::getline(config_file, line)) {
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, colon_pos);
        std::string value_str = line.substr(colon_pos + 1u);
        if (std::regex_match(value_str, integer_regex)) {
            int value = std::stoi(value_str);
            configDict[key] = value;
        }
    }
    config_file.close();

    int32_t batchSize = configDict["batchSize"];
    int32_t headNum = configDict["headNum"];
    int32_t headDim = configDict["headDim"];
    int32_t headNumDim = headNum * headDim;

    char *layOut = "TND";
    int32_t compressBlockSize = configDict["compressBlockSize"];
    int32_t compressStride = configDict["compressStride"];
    int32_t actSeqLenType = int32_t(0); // 0: count格式; 1: TND格式

    std::vector<int64_t> hostActSeqLenOptional(batchSize); // 前缀和形式
    size_t inputBySize = 0;
    inputBySize = batchSize * sizeof(int64_t);
    ReadFile(currentPath + "actSeqLenOptional.bin", inputBySize, hostActSeqLenOptional.data(), inputBySize);
    int64_t totalSeqLen = hostActSeqLenOptional[batchSize - 1];

    // nsa compress input[0]: input
    std::vector<aclFloat16> hostInputData(totalSeqLen * headNumDim);
    inputBySize = totalSeqLen * headNumDim * sizeof(aclFloat16);
    ReadFile(currentPath + "/input.bin", inputBySize, hostInputData.data(), inputBySize);

    // nsa compress input[1]: weight
    std::vector<aclFloat16> hostWeightData(compressBlockSize * headNum);
    inputBySize = compressBlockSize * headNum * sizeof(aclFloat16);
    ReadFile(currentPath + "/weight.bin", inputBySize, hostWeightData.data(), inputBySize);

    // nsa compress output[0]: output
    int compressKvNum = 0;
    int preSeqLen = 0;
    for (int b = 0; b < batchSize; b++) {
        int curSeqLen = hostActSeqLenOptional[b] - preSeqLen;
        if (curSeqLen >= compressBlockSize) {
            compressKvNum += (curSeqLen - compressBlockSize) / compressStride + 1;
        }
        preSeqLen += curSeqLen;
    }
    std::vector<aclFloat16> output_host_data(compressKvNum * headNumDim, aclFloatToFloat16(0.0));

    std::vector<int64_t> inputShape = {totalSeqLen, headNum, headDim};
    std::vector<int64_t> weightShape = {compressBlockSize, headNum};
    std::vector<int64_t> outputShape = {compressKvNum, headNum, headDim};

    void *inputDevAddr = nullptr;
    void *weightDevAddr = nullptr;
    void *outputDevAddr = nullptr;

    aclTensor *input = nullptr;
    aclTensor *weight = nullptr;
    aclIntArray *actSeqLenOptional = nullptr;
    aclTensor *output = nullptr;

    // 创建input kv aclTensor
    ret = CreateAclTensor(hostInputData, inputShape, &inputDevAddr, aclDataType::ACL_FLOAT16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建input weight_kv aclTensor
    ret = CreateAclTensor(hostWeightData, weightShape, &weightDevAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建input seq_lens aclTensor
    actSeqLenOptional = aclCreateIntArray(hostActSeqLenOptional.data(), hostActSeqLenOptional.size());
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建output compress_kv aclTensor
    ret = CreateAclTensor(output_host_data, outputShape, &outputDevAddr, aclDataType::ACL_FLOAT16, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN自定义算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 计算workspace大小并申请内存
    ret = aclnnNsaCompressGetWorkspaceSize(input, weight, actSeqLenOptional, layOut, compressBlockSize, compressStride,
                                           actSeqLenType, output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0u) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }

    // 执行算子
    ret = aclnnNsaCompress(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(aclFloat16), outputDevAddr,
                      size * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 6. 保存输出结果
    size_t outputBySize = compressKvNum * headNumDim * sizeof(aclFloat16);
    WriteFile(currentPath + "output.bin", resultData.data(), outputBySize);

    // 7. 释放资源，需要根据具体API的接口定义修改
    FreeResource(input, weight, actSeqLenOptional, output, inputDevAddr, weightDevAddr, outputDevAddr, workspaceSize,
                 workspaceAddr, deviceId, &stream);

    return 0;
}
