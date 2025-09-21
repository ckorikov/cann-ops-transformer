/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file incre_flash_attention_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_V2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_V2_H_

#include <cstdint>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/data_copy_transpose_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "../../op_host/incre_flash_attention_tiling_impl.h"

namespace optiling {

constexpr uint32_t PER_TOKEN_GROUP_MODE = 6;
constexpr uint32_t PER_TENSOR_HEAD_MODE = 2;
constexpr uint32_t PER_TOKEN_HEAD_MODE = 3;
constexpr uint32_t LSE_OUTPUT_INDEX = 1;
constexpr uint32_t PER_TOKEN_PA_MODE = 4;
constexpr uint32_t PER_TOKEN_HEAD_PA_MODE = 5;
constexpr uint32_t DIM_BB = 2;
constexpr uint32_t DIM_BNB = 3;
constexpr uint32_t DIM_PER_CHANNEL_H = 1;
constexpr uint32_t HIGH_PRECISION_ROW_INVALID = 2;
constexpr uint32_t HIGH_PERFORMANCE_ROW_INVALID = 3;
constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr uint32_t SPARSE_MODE_BAND = 4;
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;
constexpr uint32_t MASKDIM_SS = 2;
constexpr uint32_t MASKDIM_BSS = 3;
constexpr uint32_t MASKDIM_B1SS = 4;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;

class IFATilingV2 {
 public:
  IFATilingV2() = default;
  ~IFATilingV2() = default;

  ge::graphStatus DoTiling(gert::TilingContext& context);
  ge::graphStatus RunBigKernelTiling(IncreFlashAttentionContext& context, IncreFlashAttentionTilingDataV2& tilingData,
                                     bool isWorkspace = false);
  ge::graphStatus IncreFlashAttentionSetTilingData(gert::TilingContext& context,
                                                   IncreFlashAttentionTilingDataV2& tilingData);
  static ge::graphStatus ConvertContext(gert::TilingContext& context, IncreFlashAttentionContext& ifaContext);
  bool NeedRollBack() {
    return passToOldTiling_;
  }

 private:
  ge::graphStatus GetNpuInfo();
  ge::graphStatus PreProcess();
  ge::graphStatus ProcessBaseTensors();
  ge::graphStatus ProcessOptionalTensors();
  ge::graphStatus ProcessPseShift();
  ge::graphStatus ProcessAttenMask();
  ge::graphStatus ProcessAttenMaskSparsePFA();
  ge::graphStatus ProcessActualSeqLen();
  ge::graphStatus ProcessQuant2();
  ge::graphStatus ProcessDequant1();
  ge::graphStatus ProcessDequant2();
  ge::graphStatus ProcessAntiQuant();
  ge::graphStatus ProcessBlockTable();
  ge::graphStatus ProcessQPaddingSize();
  ge::graphStatus ProcessKVPaddingSize();
  ge::graphStatus VerifyQuantScale2();
  bool EnableC1V1();
  void UpdatePerfMode();
  ge::graphStatus InitInOutMode();
  ge::graphStatus KvShapePostProcess();
  ge::graphStatus CheckKvCache();
  ge::graphStatus CheckKvCacheValue(uint32_t kDimNum);
  ge::graphStatus CheckInputAntiquantFormat();
  ge::graphStatus CheckKVShape();
  ge::graphStatus CheckFormat(ge::Format format, const std::string &sName);
  ge::graphStatus CheckQKOutShape();
  ge::graphStatus CheckLse();
  ge::graphStatus CheckKVHeadNum(const gert::StorageShape *inputShape);
  ge::graphStatus CheckKeyShapeTensor(const gert::Shape& aShape, size_t idx);
  ge::graphStatus EmptyTensorProcess();

  ge::graphStatus CheckUbSpace();
  ge::graphStatus CheckPABlockSize();
  ge::graphStatus SetL2CacheFlag();

  ge::graphStatus CheckKVAntiQuantPerHead(const gert::Shape &inputParaShape);
  ge::graphStatus CheckQuant2Shape(const gert::Shape &inputParaShape);
  ge::graphStatus ProcessQuant2Dtype();
  ge::graphStatus CheckKVAntiQuantPerChannel(const gert::Shape &inputParaShape);
  ge::graphStatus CheckKVAntiQuantShapePA(const gert::Shape &inputParaShape);
  ge::graphStatus CheckKVAntiQuantParaShapeLegal(const int64_t antiquantMode, const gert::Shape &inputParaShape);
  ge::graphStatus CheckAntiQuantParam(const int64_t antiquantMode, const gert::Tensor* antiquantScaleTensor, const gert::Tensor* antiquantOffsetTensor,
                                      const gert::CompileTimeTensorDesc* antiquantScaleDesc, const gert::CompileTimeTensorDesc* antiquantOffsetDesc);
  ge::graphStatus CheckSupportQLeftPadding();
  ge::graphStatus CheckSupportKVLeftPadding();
  ge::graphStatus CheckInputFormatAndLimits();
  ge::graphStatus KeyAndValueAntiQuantParamConsistencyCheck(const gert::Tensor* keyAntiquantTensor,
                                                            const gert::Tensor* valueAntiquantTensor,
                                                            const gert::CompileTimeTensorDesc * keyAntiquantDesc,
                                                            const gert::CompileTimeTensorDesc * valueAntiquantDesc,
                                                            int64_t keyAntiquantMode, int64_t valueAntiquantMode, const std::string sName);
  bool CalcUbBmm();
  bool CalcUbSoftMax();
  bool CalcUbAttenMask();
  bool CalcUbQuant();
  bool CalcUbDeQuant();
  bool CalcUbAntiQuant();
  bool CalcUbPageAttention();
  bool CalcUbKvSplit();

  bool CheckMaskTypeAndShape(const gert::Tensor* maskShape, ge::DataType attenMaskType);
  bool CheckSparseMode(bool isDefaultSparseMode, bool enableMask);
  void SetSparseModeData(bool& isBandMode, bool enableMask, bool isDefaultSparseMode);
  bool CheckMaskCrossover(const gert::Tensor* maskShape, ge::DataType attenMaskType, bool enableMask, bool isDefaultSparseMode);
  bool CheckMaskShapeCrossSparse(const gert::Tensor* maskShape, bool isDefaultSparseMode);

  bool CanChangeToNew();
  bool ShapeEqual(const gert::Shape &aShape, const gert::Shape &bShape);
  void AdjustPABmm1Tiling(uint32_t& bmm1BaseN);
  void AdjustPABmm2Tiling() const;
  std::string GetShapeStr(const gert::Shape &aShape);

  ge::graphStatus Split();
  ge::graphStatus CalcInnerSize(uint32_t seqSize);
  ge::graphStatus SplitBN();

  std::vector<int64_t> InitSparseValidArray(const int64_t* actualLens);
  bool BalanceLoad(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                   std::vector<int64_t>& localValue, std::vector<int64_t>& sparseStartIdx);
  void InitLoadValue(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                     const std::vector<int64_t>& sparseStartIdx, std::vector<int64_t>& localValue);
  void SetSparseStartIdx(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                         uint32_t* sparseStartIdx, int64_t splitFactorSize);

  bool IsFlashDecode() const;
  void PromptFlashAttentionInitOutputSplit();
  void GetActualSeqLength(int64_t &actualSeqLengths, int64_t &actualSeqLengthsKV, uint32_t bIdx);
  void GetPreNextTokensLeftUp(int64_t actualSeqLength, int64_t actualSeqLengthKV,
                              int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
  void FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV,
                              int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
  int64_t GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength,
                            int64_t sInner, int64_t sOuter, int64_t token);
  int64_t GetCalcBlockNumsOneHead(int64_t outerBlockNums, int64_t innerBlockNums, int64_t actualSeqLength,
                                  int64_t actualSeqLengthKV, int64_t preTokensLeftUp, int64_t nextTokensLeftUp);
  int64_t GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums);
  void ComputeSplitBNSeq(std::vector<int64_t> sOuterLoopTimes, std::vector<int64_t> sInnerLoopTimes,
    double coreWightTarget);
  ge::graphStatus PromptFlashAttentionSplitBNSeq();
  ge::graphStatus SplitBN_V0();
  ge::graphStatus SplitBNS();

  bool CheckWorkSpace();
  
  bool GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType);

  ge::graphStatus CalcWorkSpace();
  ge::graphStatus CalcBlockDim();
  ge::graphStatus GenTilingKey();
  uint8_t GenHeadDimProfileVal();
  uint8_t GenAntiquantModeVal();

  ge::graphStatus FillTiling();
  void FillTilingBaseParams();
  void FillTilingSplitKV();
  void FillTilingCoreParams();
  void FillTilingSingleCoreParams();
  void FillTilingSingleCoreTensorSize();
  void FillTilingSoftmax();
  void FillTilingSoftmaxFlashTiling();
  void FillTilingTranspose();
  void FillTilingOutputParams();
  bool FillTilingBmm();  // may fail

 private:
  bool passToOldTiling_ = false;
  bool isPFAFlag_ = false;
  bool needInit_ = false;
  uint32_t numHeads_ = 0;
  uint32_t sparseMode_ = 0;
  int64_t preToken_ = 0;
  int64_t nextToken_ = 0;
  float scaleValue_ = 0;
  uint32_t numKvHeads_ = 0;
  uint32_t blockSize_ = 0;
  uint32_t innerPrecise_ = 0;
  uint32_t isRowInvalid_ = 0;
  uint32_t nNumOfQInOneGroup_ = 1;
  uint32_t msdIterNum_ = 1;
  int64_t antiquantMode_ = 0;
  uint32_t antiquantPerTensorFlag_ = 0;
  uint32_t antiquantNum_ = 2;
  uint32_t antiquantPerHeadFlag_ = 0;

  uint32_t headDim_ = 0;
  uint32_t sOfQuery_ = 0;
  uint32_t seqSize_ = 0;
  uint32_t batchSize_ = 0;
  uint32_t antiquantParaSeqSize_ = 0;
  IfaLayout inputLayout_ = IfaLayout::BSH_BSND;
  uint32_t sMax_ = 0;
  uint32_t blockTypeSize_ = 0;  // 计算中间量大小
  uint32_t kvSplitPart_ = 1;

  ge::DataType inputQType_ = ge::DT_FLOAT16;
  ge::DataType inputKvType_ = ge::DT_FLOAT16;
  ge::DataType outputType_ = ge::DT_FLOAT16;

  size_t ubSize_ = 0;
  size_t l1Size_ = 0;
  size_t l0cSize_ = 0;
  size_t l0bSize_ = 0;
  uint32_t coreNum_ = 0;
  uint32_t aicNum_ = 0;
  uint32_t aivNum_ = 0;
  IfaSocVersion socVersion_ = IfaSocVersion::SOC_ASCEND_910B;
  size_t libapiSize_ = 0;

  size_t mmResUbSize_ = 0;
  size_t bmm2ResUbSize_ = 0;

  size_t softmaxFlashTmpSize_ = 0;
  size_t softmaxTmpSize_ = 0;
  size_t softMaxSize_ = 0;

  size_t selectWithByteMaskTmpMinSize_ = 0;

  bool pseShiftFlag_ = false;
  uint32_t pseShiftTypeSize_ = NUM_BYTES_FLOAT16;
  uint32_t pseShiftBatch_ = 0U;
  uint32_t pseShiftS0_ = 0U;
  uint32_t pseShiftS1_ = 0U;

  bool attenMaskFlag_ = false;
  uint32_t attenMaskBatch_ = 1;
  uint32_t attenMaskQSize_ = 0;
  uint32_t attenMaskSize_ = 0;
  uint32_t attenMaskTypeSize_ = 0;

  bool antiQuantFlag_ = false;
  size_t antiquantUb_ = 0;
  bool kvAntiParamSplitFlag_ = false;
  bool kPerChnVPerTokFlag_ = false;
  bool antiquantParamsInPageAttentionFlag_ = false;

  bool pageAttentionFlag_ = false;
  KvCacheLayout pageAttentionKvLayoutType_ = KvCacheLayout::KV_CACHE_BSH; // pa场景下kv的shape, 0:BSH 1:BNSD
  uint32_t maxBlockNumPerSeq_ = 0;
  size_t kvPageResUbSize_ = 0;
  uint32_t totalBlockNum_ = 0;
  uint64_t needBlockNum_ = 0;

  bool batchContinuousFlag_ = true;
  std::vector<int64_t> kvListSeqLens_;

  bool actualSeqLenFlag_ = false;
  bool actualSeqLenQFlag_ = false;
  bool qPaddingSizeFlag_ = false;
  bool kvPaddingSizeFlag_ = false;

  bool quantFlag_ = false;
  size_t quantUbSize_ = 0;

  uint32_t actualLenDims_ = 0;
  uint32_t actualLenQDims_ = 0;
  uint32_t maxActualseq_ = 0;

  // flash config
  uint32_t sInnerLoopTimes_ = 0;
  uint32_t sInnerSize_ = 0;  // flash attention
  uint32_t sOuterSize_ = 0;
  uint32_t sInnerSizeTail_ = 0;
  uint32_t sInnerSizeAlign_ = 0;
  uint32_t headDimAlign_ = 0;
  // uint32_t sOuterSize_;  // flash decode s2

  bool isSplitBPolicy_ = false;
  bool splitKVFlag_ = false;

  IfaPerfMode perfMode_ = IfaPerfMode::NORMAL;
  TilingInOutMode inOutMode_ = TilingInOutMode::FP16_FP16;
  size_t workspaceSize_ = 0;

  uint32_t taskRation_ = 0;
  uint32_t usedCoreNum_ = 0;
  bool antiqMSDFlag_ = false;

  uint32_t startIdxEachCore_[MAX_CORE_NUM_REGBASE] = {};
  uint32_t coreSposStart_[MAX_CORE_NUM_REGBASE] = {};
  IncreFlashAttentionContext* context_ = nullptr;
  IncreFlashAttentionTilingData* tilingData_ = nullptr;
  bool isWorkspace_ = false;
  
  uint32_t formerCoreNum_ = 0;
  uint32_t blockSplitBn2Range_ = 0;
  uint32_t tailSplitedBatchRange_ = 0;

  uint32_t l2CacheOffFlag_ = 0;
  // softmaxLse
  bool softmaxLseFlag_ = false;
};

}  // namespace optiling 
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_H_