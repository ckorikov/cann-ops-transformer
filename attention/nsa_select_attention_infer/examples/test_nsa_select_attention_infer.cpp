/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_nsa_select_attention_infer.cpp
 * \brief
 */

 #include <iostream>
 #include <cstdio>
 #include <string>
 #include <vector>
 #include <fstream>
 #include <sys/stat.h>
 #include "acl/acl.h"
 #include "aclnnop/aclnn_nsa_select_attention_infer.h"
 
 
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
 
 template <typename T> void SaveOutResult(std::string &fileName, std::vector<int64_t> &shape, void **deviceAddr)
 {
     auto size = GetShapeSize(shape);
     std::vector<T> resultData(size, 0);
     auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                            size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
     std::ofstream file(fileName, std::ios::binary);
     // 保存文件
     file.write(static_cast<char *>((void *)resultData.data()), size * sizeof(T));
     file.close();
 }
 
 int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
 {
     // 固定写法，AscendCL初始化
     auto ret = aclInit(nullptr);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
     ret = aclrtSetDevice(deviceId);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
     ret = aclrtCreateContext(context, deviceId);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
                                   aclFinalize(); return ret);
     ret = aclrtSetCurrentContext(*context);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret);
                                   aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
     ret = aclrtCreateStream(stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
                                   aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
     return 0;
 }
 
 int ReadBinFileNNop(std::string &filePath, void *buffer, size_t bufferSize)
 {
     struct stat sBuf;
     int fileStatus = stat(filePath.data(), &sBuf);
     CHECK_RET(fileStatus == ACL_SUCCESS, LOG_PRINT("Failed to get file %s\n", filePath.c_str()); return -1);
 
     std::ifstream file;
     file.open(filePath, std::ios::binary);
     CHECK_RET(file.is_open(), LOG_PRINT("Open file failed.\n"); return -1);
 
     file.seekg(0, file.end);
     uint64_t binFileBufferLen = file.tellg();
     CHECK_RET(binFileBufferLen > 0, std::cout << "File size is 0.\n"; file.close(); return -1);
 
     file.seekg(0, file.beg);
     file.read(static_cast<char *>(buffer), binFileBufferLen);
     file.close();
     return ACL_SUCCESS;
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
 
     // 计算连续tensor的strides
     std::vector<int64_t> strides(shape.size(), 1);
     for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
         strides[i] = shape[i + 1] * strides[i + 1];
     }
 
     // 调用aclCreateTensor接口创建aclTensor
     *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                               shape.data(), shape.size(), *deviceAddr);
     return 0;
 }
 
 int CreateAclTensor(std::string &filePath, const std::vector<int64_t> &shape, int typeSize, void **deviceAddr,
                     aclDataType dataType, aclTensor **tensor)
 {
     auto size = GetShapeSize(shape) * typeSize;
     // 调用aclrtMalloc申请device侧内存
     auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
 
     // 调用aclrtMallocHost申请host侧内存
     void *binBufferHost = nullptr;
     ret = aclrtMallocHost(&binBufferHost, size);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
 
     // 读取文件
     ret = ReadBinFileNNop(filePath, binBufferHost, size);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ReadBinFileNNop failed. ERROR: %d\n", ret);
                                   (void)aclrtFreeHost(binBufferHost); return ret);
 
     // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
     ret = aclrtMemcpy(*deviceAddr, size, binBufferHost, size, ACL_MEMCPY_HOST_TO_DEVICE);
     (void)aclrtFreeHost(binBufferHost);
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
 
 void FreeResource(aclTensor *q, aclTensor *k, aclTensor *v, aclTensor *topKIndices, aclTensor *blockTableOptional,
     aclTensor *attentionOut,  void *qDeviceAddr, void *kDeviceAddr, void *vDeviceAddr, void *attentionOutDeviceAddr,
     void *topKIndicesDeviceAddr, void *blockTableDeviceAddr, uint64_t workspaceSize, void *workspaceAddr,
     int32_t deviceId, aclrtContext *context, aclrtStream *stream)
 {
     // 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
     if (q != nullptr) {
         aclDestroyTensor(q);
     }
     if (k != nullptr) {
         aclDestroyTensor(k);
     }
     if (v != nullptr) {
         aclDestroyTensor(v);
     }
     if (attentionOut != nullptr) {
         aclDestroyTensor(attentionOut);
     }
     if (topKIndices != nullptr) {
         aclDestroyTensor(topKIndices);
     }
     if (blockTableOptional != nullptr) {
         aclDestroyTensor(blockTableOptional);
     }
 
     // 释放device资源
     if (qDeviceAddr != nullptr) {
         aclrtFree(qDeviceAddr);
     }
     if (kDeviceAddr != nullptr) {
         aclrtFree(kDeviceAddr);
     }
     if (vDeviceAddr != nullptr) {
         aclrtFree(vDeviceAddr);
     }
     if (attentionOutDeviceAddr != nullptr) {
         aclrtFree(attentionOutDeviceAddr);
     }
     if (topKIndicesDeviceAddr != nullptr) {
         aclrtFree(topKIndicesDeviceAddr);
     }
     if (blockTableDeviceAddr != nullptr) {
         aclrtFree(blockTableDeviceAddr);
     }
     if (workspaceSize > 0) {
         aclrtFree(workspaceAddr);
     }
     if (stream != nullptr) {
         aclrtDestroyStream(stream);
     }
     if (context != nullptr) {
         aclrtDestroyContext(context);
     }
     aclrtResetDevice(deviceId);
     aclFinalize();
 }
 
 int main(int argc, char **argv)
 {
     // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
     // 根据自己的实际device填写deviceId
     int32_t deviceId = 0;
     aclrtContext context;
     aclrtStream stream;
     auto ret = Init(deviceId, &context, &stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
 
     // 2. 构造输入与输出，需要根据API的接口自定义构造
     // 如果需要修改shape值，需要同步修改../scripts/fa_generate_data.py中 test_nsa_select_attention_infer 分支下生成
     // query、key、value对应的shape值，并重新gen data，再执行
     int64_t batch = 1;
     int64_t s1 = 1;
     int64_t s2 = 256;
     int64_t d1 = 192;
     int64_t d2 = 128;
     int64_t g = 1;
     int64_t n2 = 1;
     int64_t blockSize = 128;
     int64_t selectBlockSize = 128;
     int64_t selectBlockCount = 2;
     int64_t numBlocks = 2;
     int64_t blockTableLength = 2;
     std::vector<int64_t> qShape = {batch, s1, n2 * g, d1};
     std::vector<int64_t> kShape = {numBlocks, blockSize, n2, d1};
     std::vector<int64_t> vShape = {numBlocks, blockSize, n2, d2};
     std::vector<int64_t> topKIndicesShape = {batch, n2, selectBlockCount};
     std::vector<int64_t> blockTableShape = {batch, blockTableLength};
     std::vector<int64_t> attentionOutShape = {batch, s1, n2 * g, d2};
     
     double scaleValue = 1.0;
     int64_t sparseMod = 0;
     char layOut[] = "BSND";
 
     void *qDeviceAddr = nullptr;
     void *kDeviceAddr = nullptr;
     void *vDeviceAddr = nullptr;
     void *topKIndicesDeviceAddr = nullptr;
     void *blockTableDeviceAddr = nullptr;
     void *attentionOutDeviceAddr = nullptr;
 
     aclTensor *q = nullptr;
     aclTensor *k = nullptr;
     aclTensor *v = nullptr;
     aclTensor *topKIndices = nullptr;
     aclTensor *attenMaskOptional = nullptr;
     aclTensor *blockTableOptional = nullptr;
     aclTensor *attentionOut = nullptr;
 
     std::vector<int64_t> actualSeqQLenVec = {1};
     std::vector<int64_t> actualSeqKvLenVec = {256};
     aclIntArray *actualSeqQLenOptional = aclCreateIntArray(actualSeqQLenVec.data(), actualSeqQLenVec.size());
     aclIntArray *actualSeqKvLenOptional = aclCreateIntArray(actualSeqKvLenVec.data(), actualSeqKvLenVec.size());//!!!这里得注意要改actualSeqKvLenVec是tensor还是intarray
 
     std::vector<uint16_t> attentionOutHostData(GetShapeSize(attentionOutShape), 0.0);
 
     uint64_t workspaceSize = 0;
     void *workspaceAddr = nullptr;
 
     if (argv == nullptr || argv[0] == nullptr) {
         LOG_PRINT("Environment error, Argv=%p, Argv[0]=%p", argv, argv == nullptr ? nullptr : argv[0]);
         return 0;
     }
     std::string exeFile(argv[0]);
     std::string currentPath = std::string(exeFile.substr(0, exeFile.rfind('/')) + "/");
     std::string qFilePath = currentPath + "query.bin";
     ret = CreateAclTensor(qFilePath, qShape, sizeof(uint16_t), &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
     CHECK_RET(ret == ACL_SUCCESS,
               FreeResource(q, k, v, topKIndices, blockTableOptional, attentionOut, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                   attentionOutDeviceAddr, topKIndicesDeviceAddr, blockTableDeviceAddr, workspaceSize, workspaceAddr,
                   deviceId, &context, &stream);
               return ret);
 
     std::string kFilePath = currentPath + "key.bin";
     ret = CreateAclTensor(kFilePath, kShape, sizeof(uint16_t), &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
     CHECK_RET(ret == ACL_SUCCESS,
               FreeResource(q, k, v, topKIndices, blockTableOptional, attentionOut, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                   attentionOutDeviceAddr, topKIndicesDeviceAddr, blockTableDeviceAddr, workspaceSize, workspaceAddr,
                   deviceId, &context, &stream);
               return ret);
 
     std::string vFilePath = currentPath + "value.bin";
     ret = CreateAclTensor(vFilePath, vShape, sizeof(uint16_t), &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
     CHECK_RET(ret == ACL_SUCCESS,
               FreeResource(q, k, v, topKIndices, blockTableOptional, attentionOut, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                   attentionOutDeviceAddr, topKIndicesDeviceAddr, blockTableDeviceAddr, workspaceSize, workspaceAddr,
                   deviceId, &context, &stream);
               return ret);
 
     std::string topKIndicesFilePath = currentPath + "topk_index.bin";
     ret = CreateAclTensor(topKIndicesFilePath, topKIndicesShape, sizeof(int32_t), &topKIndicesDeviceAddr, aclDataType::ACL_INT32, &topKIndices);
     CHECK_RET(ret == ACL_SUCCESS,
               FreeResource(q, k, v, topKIndices, blockTableOptional, attentionOut, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                   attentionOutDeviceAddr, topKIndicesDeviceAddr, blockTableDeviceAddr, workspaceSize, workspaceAddr,
                   deviceId, &context, &stream);
               return ret);
 
     std::string blockTableFilePath = currentPath + "block_table.bin";
     ret = CreateAclTensor(blockTableFilePath, blockTableShape, sizeof(int32_t), &blockTableDeviceAddr, aclDataType::ACL_INT32, &blockTableOptional);
     CHECK_RET(ret == ACL_SUCCESS,
               FreeResource(q, k, v, topKIndices, blockTableOptional, attentionOut, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                   attentionOutDeviceAddr, topKIndicesDeviceAddr, blockTableDeviceAddr, workspaceSize, workspaceAddr,
                   deviceId, &context, &stream);
               return ret);
 
     ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16,
                           &attentionOut);
     CHECK_RET(ret == ACL_SUCCESS,
               FreeResource(q, k, v, topKIndices, blockTableOptional, attentionOut, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                   attentionOutDeviceAddr, topKIndicesDeviceAddr, blockTableDeviceAddr, workspaceSize, workspaceAddr,
                   deviceId, &context, &stream);
               return ret);
 
     // 3. 调用CANN算子库API，需要修改为具体的Api名称
     aclOpExecutor *executor;
 
     // 调用aclnnNsaSelectedAttention第一段接口
     ret = aclnnNsaSelectedAttentionInferGetWorkspaceSize(
         q, k, v, topKIndices, attenMaskOptional, blockTableOptional, actualSeqQLenOptional, actualSeqKvLenOptional, layOut, n2*g, n2,
         selectBlockSize, selectBlockCount, blockSize, scaleValue, sparseMod, attentionOut, &workspaceSize, &executor);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInferGetWorkspaceSize failed. ERROR: %d\n", ret);
               FreeResource(q, k, v, topKIndices, blockTableOptional, attentionOut, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                   attentionOutDeviceAddr, topKIndicesDeviceAddr, blockTableDeviceAddr, workspaceSize, workspaceAddr,
                   deviceId, &context, &stream);
               return ret);
 
     // 根据第一段接口计算出的workspaceSize申请device内存
     if (workspaceSize > 0) {
         ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
               FreeResource(q, k, v, topKIndices, blockTableOptional, attentionOut, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                   attentionOutDeviceAddr, topKIndicesDeviceAddr, blockTableDeviceAddr, workspaceSize, workspaceAddr,
                   deviceId, &context, &stream);
               return ret);
     }
 
     // 调用aclnnNsaSelectedAttention第二段接口
     ret = aclnnNsaSelectedAttentionInfer(workspaceAddr, workspaceSize, executor, stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer failed. ERROR: %d\n", ret);
               FreeResource(q, k, v, topKIndices, blockTableOptional, attentionOut, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                   attentionOutDeviceAddr, topKIndicesDeviceAddr, blockTableDeviceAddr, workspaceSize, workspaceAddr,
                   deviceId, &context, &stream);
               return ret);
 
     // 4. （固定写法）同步等待任务执行结束
     ret = aclrtSynchronizeStream(stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer failed. ERROR: %d\n", ret);
               FreeResource(q, k, v, topKIndices, blockTableOptional, attentionOut, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                   attentionOutDeviceAddr, topKIndicesDeviceAddr, blockTableDeviceAddr, workspaceSize, workspaceAddr,
                   deviceId, &context, &stream);
               return ret);
 
     // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
     std::string attentionOutFileName = "attention_out.bin";
     SaveOutResult<short>(attentionOutFileName, attentionOutShape, &attentionOutDeviceAddr);
 
     // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改; 释放device资源
     FreeResource(q, k, v, topKIndices, blockTableOptional, attentionOut, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                 attentionOutDeviceAddr, topKIndicesDeviceAddr, blockTableDeviceAddr, workspaceSize, workspaceAddr,
                 deviceId, &context, &stream);
 
     return 0;
 }
 