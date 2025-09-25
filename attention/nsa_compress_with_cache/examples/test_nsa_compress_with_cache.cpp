/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_nsa_compress_with_cache.cpp
 */

#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_compress_with_cache.h"

#define FAILED 1

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

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
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        printf("%d %d %s\n", size, bufferSize, filePath.c_str());
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<int64_t> &shape, void **deviceAddr, aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    std::vector<T> hostData(size, 0);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}


template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 输入shape相关参数设置
    constexpr int64_t compress_block_size = 32;
    constexpr int64_t compress_stride = 16;
    constexpr int64_t heads_num = 24;
    constexpr int64_t heads_dim = 192;
    constexpr int64_t batch_size = 4;
    constexpr int64_t page_block_size = 128;
    constexpr int64_t max_seq_len = 512;
    constexpr int64_t result_len = 512;
    constexpr int64_t block_num_per_batch = max_seq_len / page_block_size;
    constexpr int64_t blocks_num = block_num_per_batch * batch_size;
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputShape = {blocks_num, page_block_size, heads_num, heads_dim};
    std::vector<int64_t> weightShape = {compress_block_size, heads_num};
    std::vector<int64_t> slotMappingShape = {batch_size};
    std::vector<int64_t> outputCacheRefShape = {result_len, heads_num, heads_dim};
    std::vector<int64_t> actSeqLenShape = {batch_size};
    std::vector<int64_t> blockTableShape = {batch_size, block_num_per_batch};

    void *inputDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *slotMappingDeviceAddr = nullptr;
    void *outputCacheRefDeviceAddr = nullptr;
    void *actSeqLenDeviceAddr = nullptr;
    void *blockTableDeviceAddr = nullptr;

    aclTensor *input = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *slotMapping = nullptr;
    aclTensor *outputCacheRef = nullptr;
    aclIntArray *actSeqLen = nullptr;
    aclTensor *blockTable = nullptr;

    std::vector<aclFloat16> inputHostData(inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3],
                                          aclFloatToFloat16(1.0));
    std::vector<aclFloat16> weightHostData(weightShape[0] * weightShape[1], aclFloatToFloat16(1.0));
    std::vector<int32_t> slotMappingHostData(slotMappingShape[0]);
    std::vector<aclFloat16> outputCacheRefHostData(outputCacheRefShape[0] * outputCacheRefShape[1] *
                                                   outputCacheRefShape[2]);
    std::vector<int64_t> actSeqLenHostData(actSeqLenShape[0]);
    std::vector<int32_t> blockTableHostData(blockTableShape[0] * blockTableShape[1]);

    size_t inputByteSize = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * sizeof(aclFloat16);
    size_t weightByteSize = weightShape[1] * weightShape[0] * sizeof(aclFloat16);
    size_t slotMappingByteSize = slotMappingShape[0] * sizeof(int32_t);
    size_t outputCacheRefByteSize =
        outputCacheRefShape[0] * outputCacheRefShape[1] * outputCacheRefShape[2] * sizeof(aclFloat16);
    size_t actSeqLenByteSize = actSeqLenShape[0] * sizeof(int64_t);
    size_t blockTableByteSize = blockTableShape[0] * blockTableShape[1] * sizeof(int32_t);

    ReadFile("./data/sample_random/input_kv.bin", inputByteSize, inputHostData.data(), inputByteSize);
    ReadFile("./data/sample_random/input_weight.bin", weightByteSize, weightHostData.data(), weightByteSize);
    ReadFile("./data/sample_random/slot_mapping.bin", slotMappingByteSize, slotMappingHostData.data(),
             slotMappingByteSize);
    ReadFile("./data/sample_random/compress_kv_cache.bin", outputCacheRefByteSize, outputCacheRefHostData.data(),
             outputCacheRefByteSize);
    ReadFile("./data/sample_random/act_seq_lens.bin", actSeqLenByteSize, actSeqLenHostData.data(), actSeqLenByteSize);
    ReadFile("./data/sample_random/input_block_table.bin", blockTableByteSize, blockTableHostData.data(),
             blockTableByteSize);

    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    ret = CreateAclTensor(slotMappingHostData, slotMappingShape, &slotMappingDeviceAddr, aclDataType::ACL_INT32,
                          &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    ret = CreateAclTensor(outputCacheRefHostData, outputCacheRefShape, &outputCacheRefDeviceAddr,
                          aclDataType::ACL_FLOAT16, &outputCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    actSeqLen = aclCreateIntArray(actSeqLenHostData.data(), actSeqLenHostData.size());

    ret = CreateAclTensor(blockTableHostData, blockTableShape, &blockTableDeviceAddr, aclDataType::ACL_INT32,
                          &blockTable);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    char layout[4] = "TND";
    int64_t actSeqLenType = 1;

    ret = aclnnNsaCompressWithCacheGetWorkspaceSize(input, weight, slotMapping, actSeqLen, blockTable, layout,
                                                    compress_block_size, compress_stride, actSeqLenType,
                                                    page_block_size, outputCacheRef, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressWithCacheGetWorkspaceSize failed. ERROR: %d\n", ret);
              return FAILED);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0UL) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
    }

    // 执行算子
    ret = aclnnNsaCompressWithCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressWithCache failed. ERROR: %d\n", ret); return FAILED);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputCacheRefShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(aclFloat16), outputCacheRefDeviceAddr,
                      size * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(outputCacheRef);
    aclDestroyIntArray(actSeqLen);
    aclDestroyTensor(blockTable);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(slotMappingDeviceAddr);
    aclrtFree(outputCacheRefDeviceAddr);
    aclrtFree(blockTableDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}