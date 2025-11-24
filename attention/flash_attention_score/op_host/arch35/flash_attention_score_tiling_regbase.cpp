/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include <alog_pub.h>
#include <tiling/tiling_api.h>
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"
#include "../../op_kernel/arch35/flash_attention_score_template_tiling_key.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h"
#include "err/ops_err.h"

using namespace Ops::Transformer::OpTiling;
namespace optiling {
namespace FA {
static const int64_t DN_D_64 = 64L;
static const int64_t DN_S1_128 = 128L;
static const uint32_t NUM_10 = 10;
static const uint32_t DIM_NUM_2 = 2;
static const uint32_t DIM_NUM_3 = 3;
static const int64_t ALIGNED_NUM_8 = 8L;
static const int64_t DROP_MASK_CAL_SIZE = 6L;
static const int64_t NUM_64 = 64L;
static const int64_t NUM_128 = 128L;
static const int64_t NUM_192 = 192L;
static const int64_t NUM_256 = 256L;
static const int64_t NUM_320 = 320L;
static const int64_t NUM_384 = 384L;
static const int64_t NUM_448 = 448L;
static const int64_t NUM_512 = 512L;
static const int64_t NUM_1024 = 1024L;
static const int64_t MIN_DN_S2 = 256L;
static const int64_t MIN_D_TO_USE_WORKSPACE = 128L;
static const int64_t DN_MIN_D_TO_USE_WORKSPACE = 192L;
static const int64_t D_TEMPLATE_SPLIT_SIZE = 64L;
static const int64_t PING_PONG_VALUE = 3L;
static const int64_t DATA_TYPE_FP32 = 4L;
static const int64_t DATA_TYPE_FP8 = 1L;
static const int64_t GM_ALIGN = 512;
static const int64_t FRACTAL_NUM = 16L;
static const int64_t PSE_DIM_NUM = 4L;
static const int64_t BYTE_BIT_NUM = 8UL;
static const int64_t HEAD_DIM_MAX_VALUE = 512L;
static const size_t PSE_INPUT_INDEX = 3UL;
static const size_t DROP_MASK_INPUT_INDEX = 4UL;
static const size_t ATTENTION_MASK_INPUT_INDEX = 6UL;
static const size_t PREFIX_INPUT_INDEX = 7UL;
static const size_t ACTUAL_SEQ_LENGTH_INPUT_INDEX = 8UL;
static const size_t ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX = 9UL;
static const size_t Q_START_IDX_INPUT_INDEX = 10UL;
static const size_t KV_START_IDX_INPUT_INDEX = 11UL;
static const size_t ATTEN_OUT_INDEX = 3UL;
static const size_t ATTENTION_MASK_DIM_NUM_4 = 4UL;
static const size_t ATTENTION_MASK_DIM_NUM_2 = 2UL;
static const int64_t HIGH_PERF_BLOCK_SIZE = 128L;
static const uint32_t PSE_ALIBI_S_SIZE = 1024;
static constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
static const int64_t ATTEN_MASK_S1_REV_INDEX = 2L;
static const int64_t ATTEN_MASK_COMPRESS_LIMIT = 2048L;
static const int64_t ATTEN_MASK_COMPRESS_PREFIX_LIMIT = 3072L;
static const int64_t SLOPE_BN_DIM_NUM = 2L;
static const int64_t SLOPE_N_DIM_NUM = 1L;
static const int64_t INVALID_ROW_SPARSE_RATIO = 6L;
static const int64_t D_Q_SCALE_INDEX = 12L;
static const int64_t D_K_SCALE_INDEX = 13L;
static const int64_t D_V_SCALE_INDEX = 14L;
static const int64_t QUERY_ROPE_INDEX = 15L;
static const int64_t KEY_ROPE_INDEX = 16L;
static const int64_t D_SCALE_DIM_NUM_4 = 4L;
static const int64_t D_SCALE_DIM_NUM_0 = 0L;
static const int64_t D_SCALE_DIM_NUM_1 = 1L;
static const int64_t D_SCALE_DIM_NUM_2 = 2L;
static const int64_t D_SCALE_DIM_NUM_3 = 3L;
static const int64_t QUANT_BLOCK_SIZE = 128L;
static const int64_t QUANT_KV_BLOCK_SIZE = 128L;

enum class LayoutType : uint8_t {
    NONE = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

enum class AttenMaskShapeType : uint8_t {
    ATTEN_B_N2_G_S1_S2 = 0,
    ATTEN_B_1_1_S1_S2 = 1,
    ATTEN_1_1_1_S1_S2 = 2,
    ATTEN_1_1_1_T_T = 99,
};

enum class PseShapeType : uint8_t {
    PSE_B_N2_G_S1_S2 = 0,
    PSE_B_N2_G_1_S2 = 1,
    PSE_B_N2_G_SLOPE,
    PSE_1_N2_G_SLOPE
};

enum class SparseMode : uint8_t {
    NO_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    PREFIX,
    PREFIX_COMPRESS,
    RIGHT_DOWN_CAUSAL_BAND,
    BAND_LEFT_UP_CAUSAL
};

enum class AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_MODE,
    PREFIX_MODE,
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
    BAND_LEFT_UP_CAUSAL_MODE
};

enum class ImplMode : uint8_t {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

enum class PseType : uint8_t {
    PSE_OUTER_MUL_ADD_TYPE = 0, // v2 default
    PSE_OUTER_ADD_MUL_TYPE, // v1 current usage
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE,
    PSE_NONE_TYPE = 9
};

enum class PseEncodeType : uint8_t {
    PSE_ENCODE_NONE = 0,
    PSE_ENCODE_ALIBI_S2_FULL = 0x11, // shape: (1024, S2)
};

enum class DTemplateType : uint32_t {
    NONALIGNED = 0,
    ALIGNED_16 = 16,
    ALIGNED_32 = 32,
    ALIGNED_48 = 48,
    ALIGNED_64 = 64,
    ALIGNED_80 = 80,
    ALIGNED_96 = 96,
    ALIGNED_128 = 128,
    ALIGNED_160 = 160,
    ALIGNED_192 = 192,
    ALIGNED_224 = 224,
    ALIGNED_256 = 256,
    ALIGNED_320 = 320,
    ALIGNED_384 = 384,
    ALIGNED_448 = 448,
    ALIGNED_512 = 512,
    DTEMPLATEBOTTOM
};

enum class STemplateType : uint32_t {
    NONALIGNED = 0,
    ALIGNED_16 = 16,
    ALIGNED_32 = 32,
    ALIGNED_64 = 64,
    ALIGNED_128 = 128,
    ALIGNED_256 = 256,
    ALIGNED_512 = 512,
    STEMPLATEBOTTOM
};

enum class ConstTemplateType : uint8_t {
    NONALIGNED128 = 0,
    ALIGNED_128 = 1,
    ONLY_128 = 2,
    LESS_THAN_128 = 3,
    ALIGNED_64 = 4,
    ALIGNED_80 = 5,
    ALIGNED_16 = 6,
    ALIGNED_256 = 7,
    ALIGNED_96 = 8,
};

struct FACompileInfoCommon {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    int64_t coreNum;
    platform_ascendc::SocVersion socVersion;
    uint32_t rsvd;
};

template <typename T>
static auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
static auto AlignDown(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return num1 / num2 * num2;
}

template <typename T>
static auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
static auto CeilDiv(const T n1, const T n2) -> T
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

template <typename T>
static auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

class FlashAttentionScoreConstTiling : public TilingBaseClass {
public:
    explicit FlashAttentionScoreConstTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~FlashAttentionScoreConstTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Reset();
        TilingBaseClass::Reset(context);
    }

protected:
    void Reset() {
        bmmDtype = matmul_tiling::DataType::DT_FLOAT;
        bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
        bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

        inputDtype = ge::DT_FLOAT16;
        inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
        calcTypeSize = inputDtypeBytes;

        tilingKeyDType = DtypeEnum::FLOAT16;

        bSize = 0LL;
        gSize = 0LL;
        dSize = 0LL;
        dSizeV = 0LL;
        dSizeRope = 0LL;
        n1Size = 0LL;
        n2Size = 0LL;
        s1Size = 0LL;
        s2Size = 0LL;
        accumS1 = 0LL;
        accumS2 = 0LL;
        bandIndex = 0LL;
        dropTotalSize = 0LL;
        realT1Size = 0LL;

        s1StrideSize = 0LL;
        s2StrideSize = 0LL;
        preTokens = std::numeric_limits<int32_t>::max();
        nextTokens = std::numeric_limits<int32_t>::max();
        sparseMode = static_cast<int64_t>(SparseMode::NO_MASK);
        pseType = static_cast<int64_t>(PseType::PSE_OUTER_ADD_MUL_TYPE);
        pseAlibiBaseS1 = 0;
        pseAlibiBaseS2 = 0;
        qStartIdx = 0;
        kvStartIdx = 0;
        keepProb = 1.0f;
        scaleValue = 1.0f;
        attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE);
        isHighPercision = true;

        s1BasicBlock = std::numeric_limits<int64_t>::max();
        s2BasicBlock = std::numeric_limits<int64_t>::max();
        dBasicBlock = std::numeric_limits<int64_t>::max();

        maxValidS2Len = 0LL;
        opName = nullptr;
        inputLayout = nullptr;
    }

    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    virtual bool GetSparseInfo(SparseEnum &sparseType);
    void EnableBandInvalidLineImplMode();
    bool SparseModeProcess(SparseEnum &sparseType);
    void SetSparseTilingInfo(SparseEnum &sparseType);
    bool SparseBandModeCheck(int64_t maxS1Val, int64_t maxS2Val, int64_t minS1Val, int64_t minS2Val,
                             SparseEnum &sparseType);
    bool PretokenAndNexttokenAdjustment(SparseEnum &sparseType);

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override = 0;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override = 0;
    // 7、保存Tiling数据，// 由于这个类中不保存TilingData，子类中需要调用这个类的PostTiling并额外设置RawTilingData的DataSize
    ge::graphStatus PostTiling() override;

    bool GetActualSeqLenData(int64_t inputIdx, std::vector<int64_t> &res, int64_t &actualLen) const;

    // 关于TilingData的校验需要在子类中实现
    virtual ge::graphStatus CheckContext();
    virtual bool AnalyzeDtype();
    bool AnalyzeAttrs();
    bool AnalyzeLayout();
    bool AnalyzeTndLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape);
    bool Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape,
                           size_t layoutLen, const gert::Shape *queryRopeShape);
    bool Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape,
                           size_t layoutLen, const gert::Shape *queryRopeShape);
    bool AnalyzeTndPseOptionalInput(PseShapeType &pseShapeType, const gert::Shape &pseShapeDims, size_t pseDimNum,
                                    int64_t pseBSize);
    bool AnalyzeGeneralPseOptionalInput(PseShapeType &pseShapeType, const gert::Shape &pseShapeDims, size_t pseDimNum,
                                        int64_t pseBSize);
    bool AnalyzePseOptionalInput();
    bool Analyze4DimAttenOptionalInput(AttenMaskShapeType &attenMaskShapeType,
                                       const gert::Shape &attenMaskStorageShape);
    bool Analyze2DimAttenOptionalInput(AttenMaskShapeType &attenMaskShapeType,
                                       const gert::Shape &attenMaskStorageShape);
    bool AnalyzeAttenOptionalInputDimNumLimit(const gert::Shape &attenMaskStorageShape,
                                              size_t attenMaskDimNum);
    bool AnalyzeAttenOptionalInput();
    bool AnalyzeDropOptionalInput();
    bool AnalyzeFp8OptionalInput();
    bool AnalyzeRopeOptionalInput();
    bool AnalyzeOptionalInput();
    virtual void CalcS1S2BasicBlock() = 0;
    virtual void CalcDBasicBlock() = 0;
    virtual void CalcDVBasicBlock();
    virtual int64_t CalcTotalSize();

    virtual ge::graphStatus SetQKVStartIdx();
    virtual void SetOutputDtype();
    virtual void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum);
    virtual void SetSparseParamsRegbase(int64_t maxCoreNum);
    virtual bool SetPseAlibiParamsRegbase();
    virtual bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx);
    virtual bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, MultiCoreParamsRegbase &multiCoreParamsRegbase,
                                   int64_t maxCoreNum);
    void SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>> &sparseValidArray,
                                 MultiCoreParamsRegbase &multiCoreParamsRegbase, int64_t maxCoreNum);
    void PrintSparseMaxMinLoadPerCore(const std::vector<int64_t> &sparseValidArray, int64_t *sparseStartIdx,
                                      int32_t validAivNum, int64_t avgLoadSize);
    bool PartitionSparseData(const std::vector<int64_t> &sparseRollingArray, int64_t sparseRollingArraySum,
                             int64_t sparseArraySize, int64_t loadMaxEachCore, std::vector<int64_t> &partitionResult);
    SparseEnum GetPrefixNList(std::ostringstream &failReason);

    uint32_t aivNum;
    uint32_t aicNum;
    platform_ascendc::SocVersion socVersion;

    matmul_tiling::DataType bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    ge::DataType inputDtype;
    int64_t inputDtypeBytes;
    int64_t calcTypeSize;

    bool isHighPercision; // fp16 high percision mode

    DtypeEnum tilingKeyDType;
    LayoutType tilingKeyLayout;
    ImplMode implMode;

    int64_t bSize;
    int64_t gSize;
    int64_t dSize;
    int64_t dSizeV;
    int64_t dSizeRope;
    int64_t n1Size;
    int64_t n2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1StrideSize; // query Shape S inner axes, for bmm1
    int64_t s2StrideSize; // key Shape S inner axes, for bmm1
    int64_t preTokens;
    int64_t nextTokens;
    int64_t s1SparseValidSize;
    int64_t s2SparseValidSize;
    int64_t sparseMode;
    int64_t pseType;
    int64_t pseAlibiBaseS1;
    int64_t pseAlibiBaseS2;
    int64_t qStartIdx;
    int64_t kvStartIdx;
    int64_t accumS1;
    int64_t dropTotalSize;
    int64_t accumS2;
    int64_t bandIndex;
    int64_t realT1Size;
    std::vector<int64_t> actualSeqLenData;
    std::vector<int64_t> actualSeqLenKvData;
    float keepProb;
    int64_t keepProbUint8;
    int64_t seed;
    int64_t offset;
    int64_t outDtype;
    float scaleValue;
    uint8_t attenMaskCompressMode;

    int64_t s1BasicBlock;
    int64_t s2BasicBlock;
    int64_t dBasicBlock;
    int64_t dVBasicBlock;

    int64_t maxValidS2Len;

    const char *templateName = "base";
    const char *opName;
    const char *inputLayout;
    const int64_t *prefixNData;

    bool isSparseValidSizeAligned = false;
    bool hasPse = false;
    bool hasAttenMask = false;
    bool hasDropOut = false;
    bool dropMaskOuter = false;
    bool regbase = false;
    bool hasRope = false;

    DTemplateType dTemplateType = DTemplateType::DTEMPLATEBOTTOM;
    DTemplateType dVTemplateType = DTemplateType::DTEMPLATEBOTTOM;
    FlashAttentionScoreSimplifiedTilingData *tilingData = context_->GetTilingData<FlashAttentionScoreSimplifiedTilingData>();
    InputParamsRegbase *inputParamsRegbase_ = &tilingData->inputParamsRegbase;
    MultiCoreParamsRegbase *multiCoreParamsRegbase_ = &tilingData->multiCoreParamsRegbase;
    DropmaskParamsRegbase *dropmaskParamsRegbase_ = &tilingData->dropmaskParamsRegbase;
};

ge::graphStatus FlashAttentionScoreConstTiling::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    size_t *workspaces = context_->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keepProbPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, preTokensPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, nextTokensPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, n1SizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    auto queryShape = context_->GetInputShape(0);
    auto queryDesc = context_->GetInputDesc(0);
    auto keyShape = context_->GetInputShape(1);
    auto attenOutShape = context_->GetOutputShape(ATTEN_OUT_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attenOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreConstTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FACompileInfoCommon *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compileInfoPtr is null."),
                   return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        socVersion = compileInfoPtr->socVersion;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        socVersion = ascendcPlatform.GetSocVersion();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }
    OP_LOGI(context_, "get platform from compileInfo. aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
              aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreConstTiling::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(0)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    switch (inputDtype) {
        case ge::DT_FLOAT16:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
            bmm1OutDtype = isHighPercision ? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
            tilingKeyDType = isHighPercision ? DtypeEnum::FLOAT16_PRECISION : DtypeEnum::FLOAT16;
            calcTypeSize = isHighPercision ? ge::GetSizeByDataType(ge::DT_FLOAT) : ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_HIFLOAT8:
            tilingKeyDType = (DtypeEnum)6; // 6 means DtypeEnum::HIFLOAT8;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            break;
        case ge::DT_FLOAT8_E5M2:
            tilingKeyDType = (DtypeEnum)4; // 4 means DtypeEnum::FLOAT8_E5M2;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            break;
        case ge::DT_FLOAT8_E4M3FN:
            tilingKeyDType = (DtypeEnum)5; // 5 means DtypeEnum::FLOAT8_E4M3;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            break;    
        case ge::DT_FLOAT:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::FLOAT32;
            isHighPercision = false;
            calcTypeSize = ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_BF16:
            bmmDtype = matmul_tiling::DataType::DT_BF16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::BFLOAT16;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            isHighPercision = false;
            break;
        default:
            OPS_REPORT_VECTOR_INNER_ERR(opName, "not support input dtype: %s for now",
                                        ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
            return false;
    }

    bmm2OutDtype = bmm1OutDtype;
    OP_LOGD(context_, "get high precision flag: %d.", isHighPercision);
    return true;
}

bool FlashAttentionScoreConstTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    inputLayout = attrs->GetAttrPointer<char>(idx++);

    preTokens = *preTokensPtr;
    nextTokens = *nextTokensPtr;
    keepProb = *keepProbPtr;
    scaleValue = *scaleValuePtr;
    n1Size = *n1SizePtr;
    if (preTokens > std::numeric_limits<int32_t>::max()) {
        OP_LOGW(context_, "preTokens[%ld] config error, should not greater than max int value."
            "preTokens will be reset max int value.", preTokens);
        preTokens = std::numeric_limits<int32_t>::max();
    } else if (preTokens < std::numeric_limits<int32_t>::min()) {
        OP_LOGW(context_, "preTokens[%ld] config error, should not less than min int value."
            "preTokens will be reset min int value.", preTokens);
        preTokens = std::numeric_limits<int32_t>::min();
    }
    if (nextTokens > std::numeric_limits<int32_t>::max()) {
        OP_LOGW(context_, "nextTokens[%ld] config error, should not greater than max int value."
            "nextTokens will be reset max int value.", nextTokens);
        nextTokens = std::numeric_limits<int32_t>::max();
    } else if (nextTokens < std::numeric_limits<int32_t>::min()) {
        OP_LOGW(context_, "nextTokens[%ld] config error, should not less than min int value."
            "nextTokens will be reset min int value.", nextTokens);
        nextTokens = std::numeric_limits<int32_t>::min();
    }
    OP_CHECK_IF(n1Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "head num is zero."), return false);
    OP_CHECK_IF(keepProb <= 0.0 || keepProb > 1.0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "keepProb value must be in range of (0, 1]."), return false);
    keepProbUint8 = static_cast<int64_t>(keepProb * UINT8_MAX);
    hasDropOut = (keepProb < 1.0f);

    implMode = ImplMode::AA_HIGH_PRECISION;
    if (attrs->GetAttrNum() > idx) {
        auto implModePtr = attrs->GetAttrPointer<uint8_t>(idx++);
        if (static_cast<ImplMode>(*implModePtr) == ImplMode::AA_INVALID_LINE_HIGH_PRECISION) {
            implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
        }
        isHighPercision = true; // use default value
    }

    if (attrs->GetAttrNum() > idx) {
        auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
        sparseMode = *sparseModePtr;
        if (sparseMode == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::BAND)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL_BAND)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_BAND_MODE);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::BAND_LEFT_UP_CAUSAL_MODE);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE);
        }
        OP_LOGD(context_, "The current value of attenMaskCompressMode is %u.", attenMaskCompressMode);
    }
    if (attrs->GetAttrNum() > idx) {
        auto pseTypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        pseType = *pseTypePtr;
        OP_CHECK_IF(pseType < 0 || pseType >= static_cast<uint8_t>(PseType::PSE_INVALID_TYPE),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "pseType value is out of range"), return false);
    }
    if (attrs->GetAttrNum() > idx) {
        auto seedPtr = attrs->GetAttrPointer<int64_t>(idx++);
        seed = *seedPtr;
    }
    if (attrs->GetAttrNum() > idx) {
        auto offsetPtr = attrs->GetAttrPointer<int64_t>(idx++);
        offset = *offsetPtr;
    }
    if (attrs->GetAttrNum() > idx) {
        auto outDtypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        outDtype = *outDtypePtr;
        OP_CHECK_IF(outDtype < 0 || outDtype >= 2,
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "outDtype value is out of range"), return false);
        outDtype = outDtype + 1; // 外部合法是0或1, 内部对应使用1和2,如果没有量化参数, 后面会刷成0, 1表示fp16, 2表示bf16
    }
    OP_LOGD(context_, "attrs: scale_value[%f] keep_prob[%f] pre_tockens[%ld] next_tockens[%ld] head_num[%ld]"
                        "input_layout[%s] inner_precise[%d] sparse_mode[%ld] pseType[%ld] seed[%ld] offset[%ld] outDtype[%ld].",
              scaleValue, keepProb, preTokens, nextTokens, n1Size, inputLayout, static_cast<int>(implMode), sparseMode, pseType,
              seed, offset, outDtype);
    return true;
}

bool FlashAttentionScoreConstTiling::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(1)->GetStorageShape();
    auto &valueShape = context_->GetInputShape(2)->GetStorageShape();

    auto queryRope = context_->GetOptionalInputShape(QUERY_ROPE_INDEX);
    bool hasQueryRope = queryRope != nullptr && queryRope->GetStorageShape().GetDimNum() != 0;
    auto keyRope = context_->GetOptionalInputShape(KEY_ROPE_INDEX);
    bool hasKeyRope = keyRope != nullptr && keyRope->GetStorageShape().GetDimNum() != 0;
    if (hasQueryRope ^ hasKeyRope) {
        OP_LOGE(opName, "query_rope and key_rope should be present or absent at the same time, check this.");
        return false;
    }
    dSizeRope = 0; // init dSizeRope
    hasRope = hasQueryRope && hasKeyRope;
    const gert::Shape *queryRopeShape = hasQueryRope ? &queryRope->GetStorageShape() : nullptr;

    size_t layoutLen = strlen(inputLayout);
    OP_LOGD(context_, "get input_layout [%s].", inputLayout);
    OP_CHECK_IF(queryShape.GetDimNum() != layoutLen || keyShape.GetDimNum() != layoutLen ||
               valueShape.GetDimNum() != layoutLen,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid layout[%s].", inputLayout), return false);
    OP_CHECK_IF(!Analyze3DimLayout(queryShape, keyShape, valueShape, layoutLen, queryRopeShape)||
               !Analyze4DimLayout(queryShape, keyShape, valueShape, layoutLen, queryRopeShape),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "get unsupported layout: %s", inputLayout), return false);
    if (s1Size > std::numeric_limits<int32_t>::max() || s2Size > std::numeric_limits<int32_t>::max()) {
        OP_LOGE(context_, "s1Size[%ld] and s2Size[%ld] config error, both should not greater than max int value.",
            s1Size, s2Size);
        return false;
    }
    OP_CHECK_IF(gSize == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSize is zero"), return false);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero"), return false);
    OP_CHECK_IF(dSize <= 0L,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "query or key dSize is not support <= 0"), return false);
    OP_CHECK_IF(dSizeV <= 0L,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "value's dSize is not support <= 0"), return false);
    OP_CHECK_IF(dSizeV > dSize,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "value's dSize is larger than query's dSize"), return false);
    OP_CHECK_IF(n1Size % n2Size != 0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "n1Size [%ld] should be a multiple of n2Size [%ld]", n1Size, n2Size),
               return false);
    return true;
}

bool FlashAttentionScoreConstTiling::GetActualSeqLenData(
    int64_t inputIdx, std::vector<int64_t> &res, int64_t &actualLen) const
{
    auto actualSeqLenTensor = context_->GetOptionalInputTensor(inputIdx);
    if (actualSeqLenTensor == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor is null pointer", templateName);
        return true;
    }
    auto &actualSeqLenShape = actualSeqLenTensor->GetShape().GetStorageShape();
    if (actualSeqLenShape.GetDimNum() != 1) {
        OP_LOGW(context_, "[%s]actualSeqLenShape is invalid %lu %ld", templateName, actualSeqLenShape.GetDimNum(),
                  actualSeqLenShape.GetDim(0));
        return true;
    }
    /* Get Data from tensor. */
    const int64_t *value = actualSeqLenTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor data is null pointer", templateName);
        return true;
    }
    int64_t seqLen = actualSeqLenShape.GetDim(0);
    try {
        res.reserve(seqLen);
    } catch (...) {
        OPS_REPORT_VECTOR_INNER_ERR(opName, "Init actual_seq_len failed, array is too long.");
        return false;
    }
    res.emplace_back(value[0]);
    actualLen++;
    for (auto i = 1; i < seqLen; ++i) {
        auto qLen = value[i] - value[i - 1];
        res.emplace_back(qLen < 0 ? 0 : qLen);
        actualLen++;
    }
    return true;
}

bool FlashAttentionScoreConstTiling::AnalyzeTndLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                                      const gert::Shape &valueShape)
{
    int64_t actualSeqQLen = 0;
    int64_t actualSeqKVLen = 0;
    int64_t t1Size = queryShape.GetDim(0);
    int64_t t2Size = keyShape.GetDim(0);
    realT1Size = t1Size;
    std::fill(actualSeqLenData.begin(), actualSeqLenData.end(), 0);
    std::fill(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0);
    if (!GetActualSeqLenData(ACTUAL_SEQ_LENGTH_INPUT_INDEX, actualSeqLenData, actualSeqQLen)) {
        OP_LOGE(opName, "Get actual_seq_qlen failed.");
        return false;
    }
    if (!GetActualSeqLenData(ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX, actualSeqLenKvData, actualSeqKVLen)) {
        OP_LOGE(opName, "Get actual_seq_kvlen failed.");
        return false;
    }
    OP_CHECK_IF(actualSeqQLen != actualSeqKVLen,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "VarLen scene, q is not equal kv."), return false);
    bSize = actualSeqQLen;
    accumS1 = std::accumulate(actualSeqLenData.begin(), actualSeqLenData.end(), 0LL);
    accumS2 = std::accumulate(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0LL);
    OP_CHECK_IF(
        t1Size < accumS1 || t2Size < accumS2,
        OPS_REPORT_VECTOR_INNER_ERR(
            opName,
            "Query T(%ld) and key T(%ld) need larger than respectively sum of seqLen(%ld) and sekvLen(%ld).",
            t1Size, t2Size, accumS1, accumS2),
        return false);
    uint32_t firstValidIndex = 0;
    uint32_t lastValidIndex = static_cast<uint32_t>(bSize - 1);
    for (int64_t i = 0; i < bSize; ++i) {
        if (actualSeqLenData[i] != 0) {
            firstValidIndex = static_cast<uint32_t>(i);
            break;
        }
    }
    for (auto i = bSize - 1; i >= 0; --i) {
        if (actualSeqLenData[i] != 0) {
            lastValidIndex = static_cast<uint32_t>(i);
            break;
        }
    }
    if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL_BAND)) {
        bandIndex = static_cast<int64_t>(lastValidIndex);
        inputParamsRegbase_->set_bandIndex(lastValidIndex);
    }
    if (sparseMode == static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)) {
        bandIndex = static_cast<int64_t>(firstValidIndex);
        inputParamsRegbase_->set_bandIndex(firstValidIndex);
    }
    s1Size = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
    s2Size = *std::max_element(actualSeqLenKvData.begin(), actualSeqLenKvData.end());
    OP_CHECK_IF(n1Size != queryShape.GetDim(1),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "head_num is [%ld], but got query dim1 [%ld].", n1Size,
                                            queryShape.GetDim(1)),
                return false);
    n2Size = keyShape.GetDim(1);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
    gSize = queryShape.GetDim(1) / n2Size;
    dSize = queryShape.GetDim(DIM_NUM_2);
    dSizeV = valueShape.GetDim(DIM_NUM_2);
    return true;
}

bool FlashAttentionScoreConstTiling::Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                                       const gert::Shape &valueShape, size_t layoutLen,
                                                       const gert::Shape *queryRopeShape)
{
    int64_t h1 = 0;
    int64_t h2 = 0;
    int64_t h3 = 0;
    int64_t hRope = 0;
    if (layoutLen == 3UL) {
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[DIM_NUM_2] == 'H') { // 2: H idx
            s1Size = queryShape.GetDim(1);
            bSize = queryShape.GetDim(0);
            s2Size = keyShape.GetDim(1);
            h1 = queryShape.GetDim(2); // 2: H idx
            h2 = keyShape.GetDim(2);   // 2: H idx
            h3 = valueShape.GetDim(2);   // 2: H idx
            if (hasRope) {
                hRope = queryRopeShape->GetDim(DIM_NUM_2);
            }
            s1StrideSize = h1;
            s2StrideSize = h2;
            inputParamsRegbase_->set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_BSH));
            tilingKeyLayout = LayoutType::LAYOUT_BSH;
        } else if (inputLayout[0] == 'S' && inputLayout[1] == 'B' && inputLayout[DIM_NUM_2] == 'H') { // 2: H idx
            s1Size = queryShape.GetDim(0);
            s2Size = keyShape.GetDim(0);
            bSize = queryShape.GetDim(1);
            h1 = queryShape.GetDim(2); // 2: H idx
            h2 = keyShape.GetDim(2);   // 2: H idx
            h3 = valueShape.GetDim(2);   // 2: H idx
            if (hasRope) {
                hRope = queryRopeShape->GetDim(DIM_NUM_2);
            }
            s1StrideSize = h1 * bSize;
            s2StrideSize = h2 * bSize;
            inputParamsRegbase_->set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_SBH));
            tilingKeyLayout = LayoutType::LAYOUT_SBH;
        } else if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[DIM_NUM_2] == 'D') {
            OP_CHECK_IF(!AnalyzeTndLayout(queryShape, keyShape, valueShape),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze tnd layout error."), return false);
            h1 = n1Size * dSize;
            h2 = n2Size * dSize;
            h3 = n2Size * dSizeV;
            if (hasRope) {
                dSizeRope = queryRopeShape->GetDim(DIM_NUM_2);
            }
            hRope = n1Size * dSizeRope;
            s1StrideSize = gSize * n2Size * dSize;
            s2StrideSize = n2Size * dSize;
            inputParamsRegbase_->set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_TND));
            tilingKeyLayout = LayoutType::LAYOUT_TND;
        } else {
            return false;
        }
        OP_CHECK_IF(h1 == 0 || h2 == 0 || h3 == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "H is zero."), return false);
        OP_CHECK_IF(h1 % n1Size != 0,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "h1 [%ld] should be a multiple of n1Size [%ld].", h1, n1Size),
                   return false);
        OP_CHECK_IF(hRope % n1Size != 0,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "hRope [%ld] should be a multiple of n1Size [%ld].", h1, n1Size),
                   return false);
        dSize = h1 / n1Size;
        gSize = h1 / h2;
        dSizeRope = hRope / n1Size;
        n2Size = h2 / dSize;
        dSizeV = h3 / n2Size;
    }

    return true;
}

bool FlashAttentionScoreConstTiling::Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                                       const gert::Shape &valueShape, size_t layoutLen,
                                                       const gert::Shape *queryRopeShape)
{
    if (layoutLen == 4UL) {
        // 2: N idx, 3: D idx
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            s1Size = queryShape.GetDim(1);
            s2Size = keyShape.GetDim(1);
            n2Size = keyShape.GetDim(2); // 2: N idx
            OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            OP_CHECK_IF(n1Size != queryShape.GetDim(2),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "head_num is [%ld], but got query dim2 [%ld].", n1Size,
                                                   queryShape.GetDim(2)),
                       return false);
            gSize = queryShape.GetDim(2) / n2Size; // 2: N idx
            dSize = queryShape.GetDim(3);          // 3: D idx
            dSizeV = valueShape.GetDim(3);          // 3: D idx
            if (hasRope) {
                dSizeRope = queryRopeShape->GetDim(DIM_NUM_3);
            }
            s1StrideSize = gSize * n2Size * dSize;
            s2StrideSize = n2Size * dSize;
            inputParamsRegbase_->set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_BSND));
            tilingKeyLayout = LayoutType::LAYOUT_BSND;
        } else if (inputLayout[0] == 'B' && inputLayout[1] == 'N' &&
                   // 2: S idx, 3: N idx
                   inputLayout[2] == 'S' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            n2Size = keyShape.GetDim(1); // 1: N idx
            OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            OP_CHECK_IF(n1Size != queryShape.GetDim(1),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "head_num is [%ld], but got query dim1 [%ld].", n1Size,
                                                   queryShape.GetDim(1)),
                       return false);
            gSize = queryShape.GetDim(1) / n2Size;
            s1Size = queryShape.GetDim(2); // 2: S idx
            s2Size = keyShape.GetDim(2);   // 2: S idx
            dSize = queryShape.GetDim(3);  // 3: D idx
            dSizeV = valueShape.GetDim(3);  // 3: D idx
            if (hasRope) {
                dSizeRope = queryRopeShape->GetDim(DIM_NUM_3);
            }
            s1StrideSize = dSize;
            s2StrideSize = dSize;
            inputParamsRegbase_->set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_BNSD));
            tilingKeyLayout = LayoutType::LAYOUT_BNSD;
        } else {
            return false;
        }
    }

    return true;
}

ge::graphStatus FlashAttentionScoreConstTiling::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid context."),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout() || !AnalyzeOptionalInput(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);

    if (hasRope && (dSize != 128 || dSizeRope != 64)) {
        OPS_REPORT_VECTOR_INNER_ERR(opName, "MLA concat only support dSize=128, dSizeRope=64.");
        return ge::GRAPH_FAILED;
    }

    inputParamsRegbase_->set_bSize(bSize);
    inputParamsRegbase_->set_n2Size(n2Size);
    inputParamsRegbase_->set_gSize(gSize);
    inputParamsRegbase_->set_s1Size(s1Size);
    inputParamsRegbase_->set_s2Size(s2Size);
    inputParamsRegbase_->set_dSize(dSize);
    inputParamsRegbase_->set_dSizeV(dSizeV);
    inputParamsRegbase_->set_dSizeRope(dSizeRope);
    inputParamsRegbase_->set_keepProb(keepProb);
    inputParamsRegbase_->set_scaleValue(scaleValue);
    inputParamsRegbase_->set_pseType(static_cast<uint32_t>(pseType));
    inputParamsRegbase_->set_keepProbUint8(keepProbUint8);
    inputParamsRegbase_->set_seed(seed);
    inputParamsRegbase_->set_offset(offset);

    OP_LOGD(context_, "input ParamsRegbase: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], keepProb[%f], scaleValue[%f],"
                        "pseType:%ld.", bSize, n2Size, gSize, s1Size, s2Size, dSize, keepProb, scaleValue, pseType);

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreConstTiling::AnalyzeTndPseOptionalInput(PseShapeType &pseShapeType,
                                                                const gert::Shape &pseShapeDims,
                                                                size_t pseDimNum, int64_t pseBSize)
{
    int64_t accumS1S2 = 0;
    for (auto i = 0; i < bSize; i++) {
        accumS1S2 += (actualSeqLenData[i] * actualSeqLenKvData[i]);
    }
    if (pseBSize == accumS2 * n1Size) {
        pseShapeType = PseShapeType::PSE_B_N2_G_1_S2;
    } else if (pseBSize == accumS1S2 * n1Size) {
        pseShapeType = PseShapeType::PSE_B_N2_G_S1_S2;
    } else if (pseDimNum == PSE_DIM_NUM && (pseShapeDims.GetDim(0) == 1 || pseShapeDims.GetDim(0) == bSize) &&
               pseShapeDims.GetDim(1) == n1Size && pseShapeDims.GetDim(DIM_NUM_2) == PSE_ALIBI_S_SIZE &&
               pseShapeDims.GetDim(DIM_NUM_3) == s2Size) {
        pseShapeType = PseShapeType::PSE_B_N2_G_S1_S2;
    } else {
        OP_LOGE(context_, "get unsupported pse shape");
        return false;
    }
    return true;
}

bool FlashAttentionScoreConstTiling::AnalyzeGeneralPseOptionalInput(PseShapeType &pseShapeType,
                                                                    const gert::Shape &pseShapeDims,
                                                                    size_t pseDimNum, int64_t pseBSize)
{
    if (pseDimNum != PSE_DIM_NUM) {
        OP_LOGE(context_, "pse dim should be 4, but got %zu", pseDimNum);
        return false;
    }
    if (pseBSize != bSize && pseBSize != 1) {
        OP_LOGE(context_, "pse batchsize should be 1 or %ld, but got %ld", bSize, pseBSize);
        return false;
    }

    int64_t pseDim1Size = pseShapeDims.GetDim(1);
    int64_t pseDim2Size = pseShapeDims.GetDim(DIM_NUM_2);
    int64_t pseDim3Size = pseShapeDims.GetDim(DIM_NUM_3);
    if (pseDim1Size == n1Size && pseDim2Size == s1Size && pseDim3Size == s2Size) { // 2: pre last axiss
        pseShapeType = PseShapeType::PSE_B_N2_G_S1_S2;
    } else if (pseDim1Size == n1Size && pseDim2Size == 1 && pseDim3Size == s2Size) {
        pseShapeType = PseShapeType::PSE_B_N2_G_1_S2;
    } else if (pseDim1Size == n1Size && pseDim2Size == static_cast<int64_t>(PSE_ALIBI_S_SIZE) &&
               pseDim3Size == s2Size) {
        if (s1Size < pseDim2Size) {
            OP_LOGE(opName, "get unsupported pse shape, the shape is [%ld, %ld, %ld, %ld]", pseBSize, pseDim1Size,
                        pseDim2Size, pseDim3Size);
            return false;
        }
        pseShapeType = PseShapeType::PSE_B_N2_G_S1_S2;
    } else {
        OP_LOGE(opName, "get unsupported pse shape, the shape is [%ld, %ld, %ld, %ld]", pseBSize, pseDim1Size,
                    pseDim2Size, pseDim3Size);
        return false;
    }
    return true;
}

bool FlashAttentionScoreConstTiling::AnalyzePseOptionalInput()
{
    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    PseShapeType pseShapeType = PseShapeType::PSE_B_N2_G_1_S2;
    auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
    if (pseShape != nullptr && pseShape->GetStorageShape().GetDimNum() != 0) {
        hasPse = true;
        auto &pseShapeDims = pseShape->GetStorageShape();
        size_t pseDimNum = pseShapeDims.GetDimNum();
        int64_t pseBSize = pseShapeDims.GetDim(0);
        if (pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            if (pseDimNum != SLOPE_BN_DIM_NUM && pseDimNum != SLOPE_N_DIM_NUM) {
                OP_LOGE(context_, "pse inner mode, unsupported pse shape");
                return false;
            }
            pseShapeType = PseShapeType::PSE_B_N2_G_SLOPE;
            if (pseDimNum == 1) {
                pseShapeType = PseShapeType::PSE_1_N2_G_SLOPE;
                pseBSize = 1;
            }
        } else if (inputParamsRegbase_->get_layoutType() == static_cast<uint8_t>(LayoutType::LAYOUT_TND)) {
            OP_CHECK_IF(!AnalyzeTndPseOptionalInput(pseShapeType, pseShapeDims, pseDimNum, pseBSize),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze tnd mode pse shape fail."),
                       return false);
        } else {
            OP_CHECK_IF(!AnalyzeGeneralPseOptionalInput(pseShapeType, pseShapeDims, pseDimNum, pseBSize),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze general pse shape fail."),
                       return false);
        }
        inputParamsRegbase_->set_pseBSize(static_cast<uint32_t>(pseBSize));
    }

    inputParamsRegbase_->set_pseShapeType(static_cast<uint8_t>(pseShapeType));
    return true;
}

bool FlashAttentionScoreConstTiling::Analyze4DimAttenOptionalInput(AttenMaskShapeType &attenMaskShapeType,
                                                                   const gert::Shape &attenMaskStorageShape)
{
    int64_t attenMaskDim0Size = attenMaskStorageShape.GetDim(0);
    int64_t attenMaskDim1Size = attenMaskStorageShape.GetDim(1);
    int64_t attenMaskDim2Size = attenMaskStorageShape.GetDim(DIM_NUM_2);
    int64_t attenMaskDim3Size = attenMaskStorageShape.GetDim(DIM_NUM_3);
    if (attenMaskDim0Size == 1 && attenMaskDim1Size == 1 && attenMaskDim2Size == s1Size &&
        attenMaskDim3Size == s2Size) {
        attenMaskShapeType = AttenMaskShapeType::ATTEN_1_1_1_S1_S2;
    } else if (attenMaskDim0Size == bSize && attenMaskDim1Size == 1 && attenMaskDim2Size == s1Size &&
                attenMaskDim3Size == s2Size) {
        attenMaskShapeType = AttenMaskShapeType::ATTEN_B_1_1_S1_S2;
    } else if (attenMaskDim0Size == bSize && attenMaskDim1Size == n1Size && attenMaskDim2Size == s1Size &&
                attenMaskDim3Size == s2Size) {
        attenMaskShapeType = AttenMaskShapeType::ATTEN_B_N2_G_S1_S2;
    } else {
        OP_LOGE(context_, "get unsupported atten_mask shape, the shape is [%ld, %ld, %ld, %ld]",
                    attenMaskDim0Size, attenMaskDim1Size, attenMaskDim2Size, attenMaskDim3Size);
        return false;
    }
    return true;
}

bool FlashAttentionScoreConstTiling::Analyze2DimAttenOptionalInput(AttenMaskShapeType &attenMaskShapeType,
                                                                   const gert::Shape &attenMaskStorageShape)
{
    int64_t attenMaskDim0Size = attenMaskStorageShape.GetDim(0);
    int64_t attenMaskDim1Size = attenMaskStorageShape.GetDim(1);
    if ((attenMaskDim0Size == s1Size && attenMaskDim1Size == s2Size) ||
        (attenMaskCompressMode != static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE))) {
        attenMaskShapeType = AttenMaskShapeType::ATTEN_1_1_1_S1_S2; // maybe [S1, S2]
    } else if (attenMaskDim0Size == accumS1 && attenMaskDim1Size == accumS2) {
        attenMaskShapeType = AttenMaskShapeType::ATTEN_1_1_1_T_T;
    } else {
        OP_LOGE(context_, "get unsupported atten_mask shape, the shape is [%ld, %ld]", attenMaskDim0Size,
                    attenMaskDim1Size);
        return false;
    }
    return true;
}

bool FlashAttentionScoreConstTiling::AnalyzeAttenOptionalInputDimNumLimit(const gert::Shape &attenMaskStorageShape,
                                                                          size_t attenMaskDimNum)
{
    if ((attenMaskCompressMode != static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE) &&
         attenMaskCompressMode != static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) &&
        ((attenMaskStorageShape.GetDim(attenMaskDimNum - ATTEN_MASK_S1_REV_INDEX) != ATTEN_MASK_COMPRESS_LIMIT) ||
         (attenMaskStorageShape.GetDim(attenMaskDimNum - 1) != ATTEN_MASK_COMPRESS_LIMIT))) {
        OP_LOGE(context_, "In the attenmask compression, please set the atten_mask_shape to [2048,2048].");
        return false;
    }
    if (attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE) &&
        ((attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - ATTEN_MASK_S1_REV_INDEX) !=
          ATTEN_MASK_COMPRESS_PREFIX_LIMIT) ||
         (attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 1) != ATTEN_MASK_COMPRESS_LIMIT))) {
        OP_LOGE(context_, "In the prefix attenmask compression, please set the atten_mask_shape to [3072,2048].");
        return false;
    }
    return true;
}

bool FlashAttentionScoreConstTiling::AnalyzeAttenOptionalInput()
{
    auto attenMaskInput = context_->GetOptionalInputDesc(ATTENTION_MASK_INPUT_INDEX);
    auto attenMaskShape = context_->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    if (attenMaskInput != nullptr && attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetDimNum() != 0) {
        hasAttenMask = true;
        auto attenMaskType = attenMaskInput->GetDataType();
        OP_CHECK_IF(attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_UINT8,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid attenMask dtype[%s], only support bool or uint8.",
                                               ge::TypeUtils::DataTypeToSerialString(attenMaskType).c_str()),
                   return false);

        inputParamsRegbase_->set_attenMaskDataType(1);
        // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
        AttenMaskShapeType attenMaskShapeType = AttenMaskShapeType::ATTEN_B_N2_G_S1_S2;
        auto &attenMaskStorageShape = attenMaskShape->GetStorageShape();
        size_t attenMaskDimNum = attenMaskStorageShape.GetDimNum();
        if (attenMaskDimNum == ATTENTION_MASK_DIM_NUM_4) {
            OP_CHECK_IF(!Analyze4DimAttenOptionalInput(attenMaskShapeType, attenMaskStorageShape),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze dim 4 attenmask shape fail."),
                       return false);
        } else if (attenMaskDimNum == ATTENTION_MASK_DIM_NUM_2) {
            OP_CHECK_IF(!Analyze2DimAttenOptionalInput(attenMaskShapeType, attenMaskStorageShape),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze dim 2 attenmask shape fail."),
                       return false);
        } else {
            OP_LOGE(context_, "atten mask dim should be 2 or 4, but got %zu", attenMaskDimNum);
            return false;
        }

        inputParamsRegbase_->set_attenMaskShapeType(static_cast<uint8_t>(attenMaskShapeType));
        OP_CHECK_IF(!AnalyzeAttenOptionalInputDimNumLimit(attenMaskStorageShape, attenMaskDimNum),
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze attenmask dim num limit error."),
                   return false);
        inputParamsRegbase_->set_attenMaskS2Size(attenMaskStorageShape.GetDim(attenMaskDimNum - 1));
    }
    return true;
}

bool FlashAttentionScoreConstTiling::AnalyzeDropOptionalInput()
{
    auto dropMaskShape = context_->GetOptionalInputShape(DROP_MASK_INPUT_INDEX);
    auto dropMaskInput = context_->GetOptionalInputDesc(DROP_MASK_INPUT_INDEX);
    if (dropMaskInput != nullptr && dropMaskShape != nullptr && dropMaskShape->GetStorageShape().GetDimNum() != 0) {
        if (!hasDropOut) {
            OP_LOGE(context_, "Dropmask parameter is invalid, please check keepProb.");
            return false;
        }
        auto dropMaskDtype = dropMaskInput->GetDataType();
        OP_CHECK_IF(dropMaskDtype != ge::DT_UINT8,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dropMask dtype[%s], only support uint8.",
                                               ge::TypeUtils::DataTypeToSerialString(dropMaskDtype).c_str()),
                   return false);
        int64_t dimNum = dropMaskShape->GetStorageShape().GetDimNum();
        int64_t dropMaskShapeSize = 1;
        int64_t shapeSize = 0;
        for (int64_t i = 0; i < dimNum; ++i) {
            int64_t dimValue = dropMaskShape->GetStorageShape().GetDim(i);
            dropMaskShapeSize *= dimValue;
        }
        if (inputParamsRegbase_->get_layoutType() == static_cast<uint8_t>(LayoutType::LAYOUT_TND)) {
            int64_t accumS1S2 = 0;
            for (auto i = 0; i < bSize; i++) {
                accumS1S2 += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeSize = accumS1S2 * n1Size;
        } else {
            shapeSize = bSize * n1Size * s1Size * s2Size;
        }
        shapeSize = AlignUp(shapeSize, BYTE_BIT_NUM) / BYTE_BIT_NUM;
        if (dropMaskShapeSize < shapeSize) {
            OP_LOGE(context_, "Input dropMask shapeSize is invalid, it should not be less than %ld, but got %ld",
                      shapeSize, dropMaskShapeSize);
            return false;
        }
        dropMaskOuter = true;
    }

    // if s2Size algined to 8, then no need dropMaskOp to transfer dropMask from bit to byte format
    inputParamsRegbase_->set_dropMaskOuter(static_cast<uint8_t>(dropMaskOuter));
    inputParamsRegbase_->set_needDropMaskOp(static_cast<uint8_t>(dropMaskOuter && s2Size % ALIGNED_NUM_8 != 0));
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        auto needDropMaskOp = (dropMaskOuter) && (s2Size % ALIGNED_NUM_8 != 0 || bSize > 1);
        inputParamsRegbase_->set_needDropMaskOp(static_cast<uint8_t>(needDropMaskOp));
    }
    return true;
}

bool FlashAttentionScoreConstTiling::AnalyzeFp8OptionalInput()
{
    auto dScaleQShape = context_->GetOptionalInputShape(D_Q_SCALE_INDEX);
    auto dScaleQInput = context_->GetOptionalInputDesc(D_Q_SCALE_INDEX);
    if (dScaleQInput != nullptr && dScaleQShape != nullptr && dScaleQShape->GetStorageShape().GetDimNum() != 0) {
        auto dScaleQDtype = dScaleQInput->GetDataType();
        OP_CHECK_IF(dScaleQDtype != ge::DT_FLOAT,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleQ dtype[%s], only support float32.",
                                               ge::TypeUtils::DataTypeToSerialString(dScaleQDtype).c_str()),
                   return false);
        int64_t dimNum = dScaleQShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(dimNum != D_SCALE_DIM_NUM_4,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleQ dimNum [%ld], only support 4 dims.", dimNum),
                return false);
        int64_t dimValue0 = dScaleQShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_0);
        int64_t dimValue1 = dScaleQShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_1);
        int64_t dimValue2 = dScaleQShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_2);
        int64_t dimValue3 = dScaleQShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_3);
        OP_CHECK_IF(dimValue0 != bSize || dimValue1 != n1Size ||
        (dimValue2 != (s1Size + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE) || dimValue3 != D_SCALE_DIM_NUM_1,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleQ dimNump[%ld][%ld][%ld][%ld], only support [B, N1, ceil(S1/128), 1]",
                dimValue0, dimValue1, dimValue2, dimValue3),
                return false);
    }

    auto dScaleKShape = context_->GetOptionalInputShape(D_K_SCALE_INDEX);
    auto dScaleKInput = context_->GetOptionalInputDesc(D_K_SCALE_INDEX);
    if (dScaleKInput != nullptr && dScaleKShape != nullptr && dScaleKShape->GetStorageShape().GetDimNum() != 0) {
        auto dScaleKDtype = dScaleKInput->GetDataType();
        OP_CHECK_IF(dScaleKDtype != ge::DT_FLOAT,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleK dtype[%s], only support float32.",
                                               ge::TypeUtils::DataTypeToSerialString(dScaleKDtype).c_str()),
                   return false);
        int64_t dimNum = dScaleKShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(dimNum != D_SCALE_DIM_NUM_4,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleK dimNum [%ld], only support 4 dims.", dimNum),
                return false);
        int64_t dimValue0 = dScaleKShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_0);
        int64_t dimValue1 = dScaleKShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_1);
        int64_t dimValue2 = dScaleKShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_2);
        int64_t dimValue3 = dScaleKShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_3);
        
        OP_CHECK_IF(dimValue0 != bSize || dimValue1 != n2Size ||
            (dimValue2 != (s2Size + QUANT_KV_BLOCK_SIZE  - 1) / QUANT_KV_BLOCK_SIZE ) || dimValue3 != D_SCALE_DIM_NUM_1,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleK dimNump[%ld][%ld][%ld][%ld], only support [B, N2, ceil(S2/128), 1]",
                    dimValue0, dimValue1, dimValue2, dimValue3),
                    return false);

    }

    auto dScaleVShape = context_->GetOptionalInputShape(D_V_SCALE_INDEX);
    auto dScaleVInput = context_->GetOptionalInputDesc(D_V_SCALE_INDEX);
    if (dScaleVInput != nullptr && dScaleVShape != nullptr && dScaleVShape->GetStorageShape().GetDimNum() != 0) {
        auto dScaleVDtype = dScaleVInput->GetDataType();
        OP_CHECK_IF(dScaleVDtype != ge::DT_FLOAT,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleV dtype[%s], only support float32.",
                                               ge::TypeUtils::DataTypeToSerialString(dScaleVDtype).c_str()),
                   return false);
        int64_t dimNum = dScaleVShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(dimNum != D_SCALE_DIM_NUM_4,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleV dimNum [%ld], only support 4 dims.", dimNum),
                return false);
        int64_t dimValue0 = dScaleVShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_0);
        int64_t dimValue1 = dScaleVShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_1);
        int64_t dimValue2 = dScaleVShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_2);
        int64_t dimValue3 = dScaleVShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_3);
        OP_CHECK_IF(dimValue0 != bSize || dimValue1 != n2Size ||
            (dimValue2 != (s2Size + QUANT_KV_BLOCK_SIZE - 1) / QUANT_KV_BLOCK_SIZE) || dimValue3 != D_SCALE_DIM_NUM_1,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleV dimNump[%ld][%ld][%ld][%ld], only support [B, N2, ceil(S2/128), 1]",
                    dimValue0, dimValue1, dimValue2, dimValue3),
                    return false);
    }
    return true;
}

bool FlashAttentionScoreConstTiling::AnalyzeOptionalInput()
{
    OP_CHECK_IF(!AnalyzePseOptionalInput() || !AnalyzeAttenOptionalInput() || !AnalyzeDropOptionalInput() ||
               !AnalyzeFp8OptionalInput(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze Optional Input error."), return false);
    OP_LOGD(context_, "hasPse: %d, hasAttenMask: %d, hasDropOut: %d, dropMaskouter %d.",
              hasPse, hasAttenMask, hasDropOut, dropMaskOuter);
    return true;
}

void FlashAttentionScoreConstTiling::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum)
{
    int64_t actualUsedCoreNum = std::min(totalSize, static_cast<int64_t>(coreNum));
    multiCoreParamsRegbase_->set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    multiCoreParamsRegbase_->set_totalSize(totalSize);
    multiCoreParamsRegbase_->set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    multiCoreParamsRegbase_->set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParamsRegbase_->get_splitFactorSize()));
}

void FlashAttentionScoreConstTiling::SetSparseParamsRegbase(int64_t maxCoreNum)
{
    if (inputParamsRegbase_->get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
        return;
    }

    inputParamsRegbase_->set_s1SparseValidSize(s1SparseValidSize);
    inputParamsRegbase_->set_s2SparseValidSize(s2SparseValidSize);

    if (inputParamsRegbase_->get_sparseType() == static_cast<uint8_t>(SparseEnum::PREFIX)) {
        std::vector<std::vector<int64_t>> sparseValidArray;
        for (int64_t bIdx = 0; bIdx < bSize; bIdx++) {
            sparseValidArray.emplace_back(std::vector<int64_t>(multiCoreParamsRegbase_->get_s1OuterSize(), 0));
            InitSparseValidArray(sparseValidArray.back(), bIdx);
        }
        SetPrefixSparseStartIdx(sparseValidArray, *multiCoreParamsRegbase_, maxCoreNum);
    } else {
        std::vector<int64_t> sparseValidArray(multiCoreParamsRegbase_->get_s1OuterSize(), 0);
        InitSparseValidArray(sparseValidArray, 0);
        SetSparseStartIdx(sparseValidArray, *multiCoreParamsRegbase_, maxCoreNum);
    }
}

ge::graphStatus FlashAttentionScoreConstTiling::DoOpTiling()
{
    OP_LOGD(context_, "try template[%s]", templateName);
    OP_CHECK_IF(dSize > HEAD_DIM_MAX_VALUE,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "query or key dSize is not in range:(0, 512]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dSizeV > HEAD_DIM_MAX_VALUE,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "value dSize is not in range:(0, 512]"), return ge::GRAPH_FAILED);
    CalcDBasicBlock();
    CalcDVBasicBlock();
    CalcS1S2BasicBlock();
    SparseEnum sparseType = SparseEnum::ALL;
    OP_CHECK_IF(!GetSparseInfo(sparseType), OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to get sparse info."),
               return ge::GRAPH_FAILED);
    SetSparseTilingInfo(sparseType);
    inputParamsRegbase_->set_implMode(static_cast<uint8_t>(implMode));
    implMode = (hasAttenMask && inputDtypeBytes != DATA_TYPE_FP32) ? implMode : ImplMode::AA_HIGH_PRECISION;
    if (!isSparseValidSizeAligned) {
        s1SparseValidSize = preTokens;
        s2SparseValidSize = nextTokens;
    }
    if (SetQKVStartIdx() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetOutputDtype();
    multiCoreParamsRegbase_->set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
    int64_t totalSize = CalcTotalSize();
    SetMultiCoreParamsRegbase(totalSize, static_cast<int64_t>(aicNum));
    SetSparseParamsRegbase(static_cast<int64_t>(aicNum));
    OP_CHECK_IF(!SetPseAlibiParamsRegbase(), OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to set pse alibi info."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreConstTiling::SetSparseTilingInfo(SparseEnum &sparseType)
{
    inputParamsRegbase_->set_attenMaskCompressMode(attenMaskCompressMode);
    inputParamsRegbase_->set_sparseType(static_cast<uint8_t>(sparseType));
    inputParamsRegbase_->set_preTokens(preTokens);
    inputParamsRegbase_->set_nextTokens(nextTokens);
}

void FlashAttentionScoreConstTiling::EnableBandInvalidLineImplMode()
{
    if (implMode == ImplMode::AA_INVALID_LINE_HIGH_PRECISION) {
        return;
    }
    // pretoken and nexttoken are already valid values (leftup vertex) after adjusted
    if (preTokens < (s1Size - s2Size) || nextTokens < 0) {
        implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
        OP_LOGI(context_, "Enable invalid line impl mode.");
        return;
    }
}

int64_t FlashAttentionScoreConstTiling::CalcTotalSize()
{
    int64_t totalSize = bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize();
    return totalSize;
}

bool FlashAttentionScoreConstTiling::PretokenAndNexttokenAdjustment(SparseEnum &sparseType)
{
    if (sparseMode == static_cast<int64_t>(SparseMode::ALL_MASK) ||
        sparseMode == static_cast<int64_t>(SparseMode::PREFIX) ||
        sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
        if (preTokens < s1Size - 1 || nextTokens < s2Size - 1) {
            OP_LOGW(context_,
                      "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                      "preTokens and nextTokens will be reset max int value.",
                      preTokens, nextTokens, sparseMode);
            preTokens = std::numeric_limits<int32_t>::max();
            nextTokens = std::numeric_limits<int32_t>::max();
        }
        sparseType = (sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) ?
                     static_cast<SparseEnum>(static_cast<uint8_t>(SparseMode::PREFIX)) :
                     static_cast<SparseEnum>(static_cast<uint8_t>(sparseMode));
    } else if (sparseMode == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) {
        if (preTokens != s1Size || nextTokens != 0) {
            OP_LOGW(context_,
                      "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                      "preTokens will be reset max int value and nextTokens will be reset 0.",
                      preTokens, nextTokens, sparseMode);
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
        }
        sparseType = SparseEnum::CAUSAL;
    } else if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        if (s1Size == s2Size) {
            if (preTokens != s1Size || nextTokens != 0) {
                OP_LOGW(context_,
                          "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                          "preTokens will be reset max int value and nextTokens will be reset 0.",
                          preTokens, nextTokens, sparseMode);
                preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
                nextTokens = 0;
            }
            sparseType = SparseEnum::CAUSAL;
        } else {
            // unequal S, change to band
            preTokens = s1Size;
            nextTokens = s2Size - s1Size;
            OP_LOGD(context_,
                      "Unequal s, sparseType rightDownCasual reset to band, and reset preTokens[%ld] "
                      "and nextTokens[%ld].",
                      preTokens, nextTokens);
            sparseType = SparseEnum::BAND;
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = preTokens;
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            isSparseValidSizeAligned = true;
        }
    } else if (sparseMode == static_cast<int64_t>(SparseMode::BAND)) {
        // unequal s, pretoken and nexttoken count from rigthDown vertex, need to change to leftUp vertex
        if (s1Size != s2Size) {
            preTokens = s1Size - s2Size + preTokens;
            nextTokens = s2Size - s1Size + nextTokens;
        }
        if (!SparseBandModeCheck(s1Size, s2Size, s1Size, s2Size, sparseType)) {
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreConstTiling::SparseBandModeCheck(int64_t maxS1Val, int64_t maxS2Val, int64_t minS1Val,
                                                         int64_t minS2Val, SparseEnum &sparseType)
{
    int64_t oriPreTokens = (sparseMode == static_cast<int64_t>(SparseMode::BAND)) ?
                           (preTokens + s2Size - s1Size) : preTokens;
    int64_t oriNextTokens = (sparseMode == static_cast<int64_t>(SparseMode::BAND)) ?
                            (nextTokens + s1Size - s2Size) : nextTokens;
    if (preTokens >= 0 && nextTokens >= 0) {
        if (preTokens >= maxS1Val && nextTokens >= maxS2Val) {
            OP_LOGW(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error, should not both greater than maxS1Val[%ld] "
                      "maxS2Val[%ld].",
                      oriPreTokens, oriNextTokens, maxS1Val, maxS2Val);
            return true;
        }
        s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
        s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
        isSparseValidSizeAligned = true;
        sparseType = SparseEnum::BAND;
        // check need to enable AA_INVALID_LINE_HIGH_PRECISION
        EnableBandInvalidLineImplMode();
        return true;
    }

    if (preTokens < 0 && nextTokens < 0) {
        OP_LOGE(context_, "PreTokens[%ld] and nextTokens[%ld] config error, there is no valid data block.",
                  oriPreTokens, oriNextTokens);
        return false;
    }

    if (preTokens < 0 && nextTokens >= 0) {
        int64_t absPreTokens = std::abs(preTokens);
        if (nextTokens >= absPreTokens && absPreTokens < minS2Val) {
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), 0L);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            isSparseValidSizeAligned = true;
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OP_LOGE(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error with S1[%ld], there is no valid data block.",
                      oriPreTokens, oriNextTokens, minS1Val);
            return false;
        }
    }

    if (preTokens >= 0 && nextTokens < 0) {
        int64_t absNextTokens = std::abs(nextTokens);
        if (absNextTokens <= preTokens && absNextTokens < minS1Val) {
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), 0L);
            isSparseValidSizeAligned = true;
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OP_LOGE(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error with S2[%ld], there is no valid data block.",
                      oriPreTokens, oriNextTokens, minS2Val);
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreConstTiling::SparseModeProcess(SparseEnum &sparseType)
{
    if (!PretokenAndNexttokenAdjustment(sparseType)) {
        return false;
    }

    if (sparseMode == static_cast<int64_t>(SparseEnum::PREFIX) || sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
        std::ostringstream failReason;
        sparseType = GetPrefixNList(failReason);
        if (sparseType != SparseEnum::PREFIX && sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
            OP_LOGE(context_, "[%s] %s.", templateName, failReason.str().c_str());
            return false;
        }

        if (sparseType == SparseEnum::PREFIX && sparseMode == static_cast<int64_t>(SparseMode::PREFIX) &&
            inputParamsRegbase_->get_attenMaskShapeType() !=
                static_cast<uint8_t>(AttenMaskShapeType::ATTEN_B_N2_G_S1_S2) &&
            inputParamsRegbase_->get_attenMaskShapeType() !=
                static_cast<uint8_t>(AttenMaskShapeType::ATTEN_B_1_1_S1_S2) && bSize != 1) {
            OP_LOGE(context_, "Prefix mode get invalid atten_mask shape, should be [BNSS] or [B1SS].");
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreConstTiling::GetSparseInfo(SparseEnum &sparseType)
{
    OP_LOGD(context_, "check sparse info: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], hasAttenMask[%d].",
              preTokens, nextTokens, s1Size, s2Size, hasAttenMask);
    if (sparseMode > static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
        OP_LOGE(context_, "Not support sparse mode of %ld.", sparseMode);
        return false;
    }

    if (!hasAttenMask) {
        return true;
    }

    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        return true;
    }

    if (sparseMode == static_cast<int64_t>(SparseMode::NO_MASK)) {
        if (preTokens >= s1Size && nextTokens == 0) {
            sparseType = SparseEnum::CAUSAL;
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
        } else {
            if (preTokens >= s1Size && nextTokens >= s2Size) {
                return true;
            }
            if (!SparseBandModeCheck(s1Size, s2Size, s1Size, s2Size, sparseType)) {
                return false;
            }
        }
    } else {
        if (!SparseModeProcess(sparseType)) {
            return false;
        }
    }
    return true;
}


bool FlashAttentionScoreConstTiling::InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx)
{
    OP_CHECK_IF(sparseValidArray.empty(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Sparse valid array size should be larger than 0."), return false);
    uint8_t sparseType = inputParamsRegbase_->get_sparseType();
    if (sparseType == static_cast<uint8_t>(SparseEnum::PREFIX)) {
        for (int64_t i = 0; i < static_cast<int64_t>(sparseValidArray.size()); i++) {
            int64_t s2IgnoredEndLen =
                inputParamsRegbase_->get_s1Size() - s1BasicBlock * (i + 1);
            int64_t s2EndLen = 0;
            s2IgnoredEndLen = std::max(static_cast<int64_t>(0), s2IgnoredEndLen);
            if (inputParamsRegbase_->get_s2Size() > s2IgnoredEndLen) {
                s2EndLen = inputParamsRegbase_->get_s2Size() - s2IgnoredEndLen;
                s2EndLen = std::max(s2EndLen, prefixNData[bIdx]);
            } else {
                s2EndLen = inputParamsRegbase_->get_s2Size();
                s2EndLen = std::min(s2EndLen, prefixNData[bIdx]);
            }

            s2EndLen = std::min(s2EndLen, inputParamsRegbase_->get_s2Size());
            sparseValidArray[i] = CeilDivision(s2EndLen, s2BasicBlock);
        }
    } else {
        int64_t s2BlkNum = CeilDivision(s2Size, s2BasicBlock);
        int64_t validS1Size = CeilDivision(s1SparseValidSize, s1BasicBlock);
        int64_t validS2Size = CeilDivision(s2SparseValidSize, s2BasicBlock);
        for (int64_t i = 0; i < static_cast<int64_t>(sparseValidArray.size()); i++) {
            int64_t reduceBlk =
                (i < validS1Size) ? 0 : (CeilDivision((i + 1) * s1BasicBlock - s1SparseValidSize, s2BasicBlock) - 1);
            int64_t addBlk =
                std::min(s2BlkNum - validS2Size,
                         CeilDivision((i + 1) * s1BasicBlock + s2SparseValidSize, s2BasicBlock) - validS2Size);
            int64_t validBlockNum = validS2Size - reduceBlk + addBlk;
            sparseValidArray[i] = validBlockNum > 0 ? validBlockNum : INVALID_ROW_SPARSE_RATIO;
            maxValidS2Len = std::max(sparseValidArray[i] * s1BasicBlock, maxValidS2Len);
        }
    }
    return true;
}

bool FlashAttentionScoreConstTiling::PartitionSparseData(const std::vector<int64_t> &sparseRollingArray,
                                                        int64_t sparseRollingArraySum, int64_t sparseArraySize,
                                                        int64_t loadMaxEachCore, std::vector<int64_t> &partitionResult)
{
    OP_CHECK_IF(partitionResult.empty(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "partitionResult size should be larger than 0."), return false);

    OP_CHECK_IF(sparseRollingArraySum <= 0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "sparseRollingArraySum should be larger than 0."), return false);
    int64_t s1OuterCutEachCore = loadMaxEachCore / sparseRollingArraySum;
    int64_t s1OuterLoadEachCore = s1OuterCutEachCore * sparseRollingArraySum;
    int64_t s1OuterNumEachCore = s1OuterCutEachCore * sparseRollingArray.size();

    int64_t targetCoreNum = partitionResult.size();
    int64_t coreIdx = 0;
    int64_t rollingIdx = 0;
    int64_t loadSize = s1OuterLoadEachCore;
    partitionResult[0] = 0;
    for (int64_t i = s1OuterNumEachCore; i < sparseArraySize; i++, rollingIdx++) {
        rollingIdx = (static_cast<uint64_t>(rollingIdx) >= sparseRollingArray.size()) ? 0 : rollingIdx;
        int64_t loadNext = sparseRollingArray[rollingIdx];
        bool needOneMoreCore = (loadSize + loadNext) > loadMaxEachCore;
        if (needOneMoreCore && coreIdx >= (targetCoreNum - 1)) {
            return false;
        }

        if (needOneMoreCore) {
            partitionResult[++coreIdx] = i;
            i += s1OuterNumEachCore;
            i--;
            rollingIdx--;
            loadSize = s1OuterLoadEachCore;
            continue;
        }

        loadSize += loadNext;
    }

    std::fill(partitionResult.begin() + coreIdx + 1, partitionResult.end(), sparseArraySize);
    return true;
}

void FlashAttentionScoreConstTiling::SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>> &sparseValidArray,
                                                             MultiCoreParamsRegbase &multiCoreParamsRegbase, int64_t maxCoreNum)
{
    int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParamsRegbase.get_coreNum()), maxCoreNum);
    int64_t totalSize = multiCoreParamsRegbase.get_totalSize(); // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParamsRegbase.get_sparseStartIdxPtr();
    for (int64_t idx = 0; idx < maxCoreNum; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }
    if (totalSize <= validAivNum) {
        int64_t idx = 0;
        for (; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }
        for (; idx < validAivNum; ++idx) {
            sparseStartIdx[idx] = totalSize;
        }
        return;
    }

    int64_t loadTotal = 0;
    /* Need to adapt when we split b. */
    for (int64_t i = 0; i < bSize; i++) {
        loadTotal += std::accumulate(sparseValidArray[i].begin(), sparseValidArray[i].end(), 0LL);
    }
    int64_t n2G = n2Size * gSize;
    loadTotal *= n2G;

    auto loadEachCoreExpect = CeilDivision(loadTotal, validAivNum);
    int64_t s1OuterSize = multiCoreParamsRegbase_->get_s1OuterSize();
    int64_t tempBlock = 0;
    int64_t coreIdx = 0;
    int64_t loadStartIdx = 0;

    for (int64_t bNGS1Index = 0; bNGS1Index < bSize * n2G * s1OuterSize; ++bNGS1Index) {
        int64_t bIdx = bNGS1Index / (n2G * s1OuterSize);
        if (s1OuterSize == 0) {
            continue;
        }
        int64_t s1Idx = bNGS1Index % s1OuterSize;
        auto currBlockNum = sparseValidArray[bIdx][s1Idx];
        if (tempBlock >= loadEachCoreExpect) {
            if ((tempBlock + currBlockNum - loadEachCoreExpect) >= (loadEachCoreExpect - (tempBlock))) {
                /* 不累加当前block */
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = bNGS1Index;
                // 下一个核使用当前这个S1
                tempBlock = currBlockNum;
            } else {
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = (bNGS1Index + 1);
                tempBlock = 0;
            }
        } else {
            tempBlock += currBlockNum;
        }
    }

    if (tempBlock != 0) {
        sparseStartIdx[coreIdx++] = loadStartIdx;
    }

    /* 将没用到的核的start index置为最大值 */
    for (; coreIdx < maxCoreNum; ++coreIdx) {
        sparseStartIdx[coreIdx] = totalSize;
    }
}

bool FlashAttentionScoreConstTiling::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                                                       MultiCoreParamsRegbase &multiCoreParamsRegbase, int64_t maxCoreNum)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParamsRegbase.get_coreNum()), maxCoreNum);
    int64_t totalSize = multiCoreParamsRegbase.get_totalSize(); // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParamsRegbase.get_sparseStartIdxPtr();
    for (int64_t idx = 0; idx < maxCoreNum; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }

    if (totalSize <= validAivNum) {
        for (int64_t idx = 0; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }

        return true;
    }

    // Minimize the max load each core to find a load balance result.
    // The range of max load each core is (loadEachCoreLowerBound, loadEachCoreUpperBound].
    std::vector<int64_t> partitionResult(validAivNum, totalSize);
    std::vector<int64_t> lastValidPartitionResult(validAivNum, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
    int64_t loadTotal = sparseArraySum * (totalSize / sparseValidArray.size());
    OP_CHECK_IF(validAivNum <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "validAivNum should be larger than 0."),
               return false);
    int64_t loadEachCoreLowerBound = loadTotal / validAivNum - 1;
    int64_t loadEachCoreUpperBound =
        CeilDivision(loadTotal, validAivNum) + (*std::max_element(sparseValidArray.begin(), sparseValidArray.end()));
    while (loadEachCoreLowerBound + 1 < loadEachCoreUpperBound) {
        int64_t loadMax = loadEachCoreLowerBound + (loadEachCoreUpperBound - loadEachCoreLowerBound) / 2;
        if ((loadMax * validAivNum >= loadTotal) &&
            PartitionSparseData(sparseValidArray, sparseArraySum, totalSize, loadMax, partitionResult)) {
            loadEachCoreUpperBound = loadMax;
            lastValidPartitionResult.swap(partitionResult);
            continue;
        }
        loadEachCoreLowerBound = loadMax;
    }

    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        sparseStartIdx[idx] = lastValidPartitionResult[idx];
    }

    if (AlogCheckDebugLevel(OP, DLOG_DEBUG) == 1) {
        PrintSparseMaxMinLoadPerCore(sparseValidArray, sparseStartIdx, validAivNum,
                                     CeilDivision(loadTotal, validAivNum));
    }
    return true;
}

void FlashAttentionScoreConstTiling::PrintSparseMaxMinLoadPerCore(const std::vector<int64_t> &sparseValidArray,
                                                                 int64_t *sparseStartIdx, int32_t validAivNum,
                                                                 int64_t avgLoadSize)
{
    int64_t maxLoadSize = 0;
    int64_t minLoadSize = std::numeric_limits<int64_t>::max();
    int64_t totalSize = multiCoreParamsRegbase_->get_totalSize();
    int64_t s1OuterSize = multiCoreParamsRegbase_->get_s1OuterSize();
    if (s1OuterSize == 0) {
        return;
    }
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t startIdx = sparseStartIdx[idx];
        int64_t endIdx = totalSize;
        if (idx + 1 < validAivNum) {
            endIdx = sparseStartIdx[idx + 1];
        }

        if (startIdx >= endIdx) {
            minLoadSize = 0;
            break;
        }

        int64_t s1OuterStartIdx = startIdx % s1OuterSize;
        int64_t s1OuterEndIdx = endIdx % s1OuterSize;
        int64_t loadSize = 0;
        if (s1OuterEndIdx > s1OuterStartIdx) {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx,
                                       sparseValidArray.begin() + s1OuterEndIdx, 0L);
        } else {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx, sparseValidArray.end(), 0L);
            loadSize = std::accumulate(sparseValidArray.begin(), sparseValidArray.begin() + s1OuterEndIdx, loadSize);
        }

        int64_t s1OuterLoop = (endIdx / s1OuterSize) - (startIdx / s1OuterSize);
        if (s1OuterLoop > 1) {
            if (s1OuterEndIdx > s1OuterStartIdx) {
                loadSize += s1OuterLoop * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0L);
            } else {
                loadSize += (s1OuterLoop - 1) * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0L);
            }
        }

        maxLoadSize = std::max(maxLoadSize, loadSize);
        minLoadSize = std::min(minLoadSize, loadSize);
    }

    OP_LOGD(context_, "[%s]each core load: max[%ld], min[%ld], avg[%ld]", templateName, maxLoadSize, minLoadSize,
              avgLoadSize);
}


SparseEnum FlashAttentionScoreConstTiling::GetPrefixNList(std::ostringstream &failReason)
{
    auto prefixN = context_->GetOptionalInputTensor(PREFIX_INPUT_INDEX);
    if (prefixN == nullptr) {
        OP_LOGW(context_, "[%s]prefixN is null pointer while sparse mode is prefix", templateName);
        failReason << "prefixN is null pointer while sparse mode is prefix";
        return SparseEnum::ALL;
    }

    auto &prefixShape = prefixN->GetShape().GetStorageShape();
    if (prefixShape.GetDimNum() != 1) {
        OP_LOGW(context_, "[%s] prefixN shape is invalid, DimNum should be 1, but it is %lu.", templateName,
                  prefixShape.GetDimNum());
        failReason << "prefixN shape is invalid, DimNum should be 1, but it is " << prefixShape.GetDimNum();
        return SparseEnum::ALL;
    }
    if (prefixShape.GetDim(0) != bSize) {
        OP_LOGW(context_, "[%s] prefixN is invalid, it should be the same size as bSize[%ld], but it is %ld.",
                  templateName, bSize, prefixShape.GetDim(0));
        failReason << "prefixN is invalid, it should be the same size as bSize[" << bSize
                   << "], but it is " << prefixShape.GetDim(0);
        return SparseEnum::ALL;
    }
    /* Get Data from tensor. */
    prefixNData = prefixN->GetData<int64_t>();
    if (prefixNData == nullptr) {
        OP_LOGW(context_, "[%s]prefixN data is null pointer", templateName);
        failReason << "prefixN data is null pointer";
        return SparseEnum::ALL;
    }

    int64_t nMin = ((s2Size - s1Size) > 0) ? (s2Size - s1Size) : 0;
    for (int64_t i = 0; i < bSize; ++i) {
        if (prefixNData[i] < nMin || prefixNData[i] > s2Size) {
            OP_LOGW(context_, "[%s] batch[%ld] prefixN=%ld is invalid, should be in range of [%ld, %ld]",
                      templateName, i, prefixNData[i], nMin, s2Size);
            failReason << "batch[" << i << "] prefixN=" << prefixNData[i] << " is invalid, should be in range of ["
                       << nMin << ", " << s2Size << "]";
            return SparseEnum::ALL;
        }

        if (s1Size > s2Size && prefixNData[i] == 0) {
            implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
            OP_LOGD(context_, "Enable invalid line impl mode.");
        }
    }

    return SparseEnum::PREFIX;
}
void FlashAttentionScoreConstTiling::SetOutputDtype() {
    if (inputDtype != ge::DT_FLOAT8_E5M2 && inputDtype != ge::DT_FLOAT8_E4M3FN && inputDtype != ge::DT_HIFLOAT8) {
        outDtype = 0;
    }
}

ge::graphStatus FlashAttentionScoreConstTiling::SetQKVStartIdx() {
    inputParamsRegbase_->set_qStartIdx(0);
    inputParamsRegbase_->set_kvStartIdx(0);
    auto qStartIdxTensor = context_->GetOptionalInputTensor(Q_START_IDX_INPUT_INDEX);
    if (qStartIdxTensor != nullptr) {
        auto &qStartIdxShape = qStartIdxTensor->GetShape().GetStorageShape();
        if (qStartIdxShape.GetDimNum() >= 1 && qStartIdxShape.GetDim(0) != 0) {
            const int64_t *value = qStartIdxTensor->GetData<int64_t>();
            if (value != nullptr) {
                qStartIdx = value[0];
                inputParamsRegbase_->set_qStartIdx(qStartIdx);
                OP_LOGD(context_, "[%s] SetQKVStartIdx qStartIdx:%ld", templateName, qStartIdx);
            }
        }
    }

    auto kvStartIdxTensor = context_->GetOptionalInputTensor(KV_START_IDX_INPUT_INDEX);
    if (kvStartIdxTensor != nullptr) {
        auto &kvStartIdxShape = kvStartIdxTensor->GetShape().GetStorageShape();
        if (kvStartIdxShape.GetDimNum() >= 1 && kvStartIdxShape.GetDim(0) != 0) {
            const int64_t *kvValue = kvStartIdxTensor->GetData<int64_t>();
            if (kvValue != nullptr) {
                kvStartIdx = kvValue[0];
                inputParamsRegbase_->set_kvStartIdx(kvStartIdx);
                OP_LOGD(context_, "[%s] SetQKVStartIdx kvStartIdx:%ld", templateName, kvStartIdx);
            }
        }
    }
    // 当kvStartIdx - qStartIdx超出范围后，由于编译器不支持大数值类型转换，kernel侧int_64转float类型时可能发生截断。
    OP_CHECK_IF(kvStartIdx - qStartIdx > INT32_MAX || kvStartIdx - qStartIdx < INT32_MIN, OPS_REPORT_VECTOR_INNER_ERR(opName,
        "kvStartIdx - qStartIdx should >= %d and <= %d, but qStartIdx = %ld, kvStartIdx = %ld.", INT32_MIN, INT32_MAX, qStartIdx, kvStartIdx),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreConstTiling::SetPseAlibiParamsRegbase()
{
    inputParamsRegbase_->set_pseEncodeType(static_cast<uint8_t>(PseEncodeType::PSE_ENCODE_NONE));
    auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
    if (pseShape == nullptr) {
        return true;
    }
    auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2);
    auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);
    if (pseS1Size == PSE_ALIBI_S_SIZE && s1Size > PSE_ALIBI_S_SIZE && pseS2Size == s2Size) {
        if (s1Size != s2Size) {
            OPS_REPORT_VECTOR_INNER_ERR(opName, "Pse alibi only support same S1 S2 when S1 lager than 1024");
            return false;
        }
    }
    return true;
}

ge::graphStatus FlashAttentionScoreConstTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreConstTiling::CalcDVBasicBlock() {
    dVBasicBlock = AlignUp(dSizeV, D_TEMPLATE_SPLIT_SIZE);
    if (dTemplateType == DTemplateType::ALIGNED_192 && hasRope) {
        dVTemplateType = DTemplateType::ALIGNED_128;
    } else {
        dVTemplateType = dTemplateType;
    }
}


ge::graphStatus FlashAttentionScoreConstTiling::PostTiling()
{
    auto blockDim = CalcTschBlockDim(multiCoreParamsRegbase_->get_coreNum() * 2, aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (inputParamsRegbase_->get_needDropMaskOp() == 1) {
        blockDim = CalcTschBlockDim(aivNum, aicNum, aivNum);
        context_->SetBlockDim(blockDim);

        int64_t shapeTotalSize = bSize * n2Size * gSize * s1Size * s2Size;
        auto layoutType = inputParamsRegbase_->get_layoutType();
        if (layoutType == static_cast<uint8_t>(LayoutType::LAYOUT_TND)) {
            for (auto i = 0; i < bSize; i++) {
                dropTotalSize += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeTotalSize = n2Size * gSize * dropTotalSize;
        }
        shapeTotalSize = AlignUp(shapeTotalSize, GM_ALIGN);
        workspaces[0] += static_cast<size_t>(shapeTotalSize);
    }

    OP_LOGD(opName, "[%s]tiling data: %s", templateName, GetTilingDataDebugStr().c_str());

    return ge::GRAPH_SUCCESS;
}


class FlashAttentionScoreTilingS1S2Const : public FlashAttentionScoreConstTiling {
public:
    explicit FlashAttentionScoreTilingS1S2Const(gert::TilingContext *context) :
        FlashAttentionScoreConstTiling(context)
    {
        this->templateName = "S1S2Const";
        this->regbase = true;
    }
    ~FlashAttentionScoreTilingS1S2Const() override = default;

protected:
    STemplateType s1TemplateType = STemplateType::STEMPLATEBOTTOM;
    STemplateType s2TemplateType = STemplateType::STEMPLATEBOTTOM;

    ge::graphStatus CheckContext() override {
        FlashAttentionScoreConstTiling::CheckContext();
        return ge::GRAPH_SUCCESS;
    }

    int64_t CalcTotalSize() override {
        int64_t totalSize = bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize();
        if (totalSize < aicNum && implMode != ImplMode::AA_INVALID_LINE_HIGH_PRECISION && !hasRope &&
            inputDtypeBytes != DATA_TYPE_FP32 && inputDtypeBytes != DATA_TYPE_FP8 && dBasicBlock <= NUM_256) {
            if (s2Size > NUM_1024 && !hasAttenMask && !hasPse && !hasDropOut) {
                s2TemplateType = STemplateType::ALIGNED_256;
                s2BasicBlock = NUM_256;
            }
            s1TemplateType = STemplateType::ALIGNED_64;
            s1BasicBlock = NUM_64;
            multiCoreParamsRegbase_->set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
            totalSize = bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize();
        }
        return totalSize;
    }

    void CalcDBasicBlock() override {
        /* 先确定D的基本块，确定的逻辑是按照64来分档 */
        dBasicBlock = AlignUp(dSize + dSizeRope, D_TEMPLATE_SPLIT_SIZE);
        switch (dBasicBlock) {
            case NUM_64:
                dTemplateType = DTemplateType::ALIGNED_64;
                break;
            case NUM_128:
                dTemplateType = DTemplateType::ALIGNED_128;
                break;
            case NUM_192:
                dTemplateType = DTemplateType::ALIGNED_192;
                break;
            case NUM_256:
                dTemplateType = DTemplateType::ALIGNED_256;
                break;
            case NUM_320:
            case NUM_384:
            case NUM_448:
            case NUM_512:
                dTemplateType = DTemplateType::ALIGNED_512;
                break;
            default:
                dTemplateType = DTemplateType::DTEMPLATEBOTTOM;
        }
    }

    void CalcS1S2BasicBlock() override
    {
        /* s2 = 64 && d == 64使能dn优化 */
        if (dSize == DN_D_64 && dSizeV == DN_D_64 &&
            (s1Size % DN_S1_128 == 0) &&
            (s2Size % MIN_DN_S2 == 0) &&
            !hasAttenMask && !hasPse && !hasDropOut && (inputDtypeBytes != DATA_TYPE_FP32) && 
            (inputDtypeBytes != DATA_TYPE_FP8) && !hasRope) {
            s1TemplateType = STemplateType::ALIGNED_128;
            s2TemplateType = STemplateType::ALIGNED_256;
            s1BasicBlock = NUM_128;
            s2BasicBlock = NUM_256;
        } else if (dSize > NUM_256) {
            if (inputDtypeBytes == DATA_TYPE_FP32) {
                s1TemplateType = STemplateType::ALIGNED_64;
                s1BasicBlock = NUM_64;
            } else {
                s1TemplateType = STemplateType::ALIGNED_128;
                s1BasicBlock = NUM_128;
            }
            s2TemplateType = STemplateType::ALIGNED_128;
            s2BasicBlock = NUM_128;
        } else {
            s1TemplateType = STemplateType::ALIGNED_128;
            s2TemplateType = STemplateType::ALIGNED_128;
            s1BasicBlock = NUM_128;
            s2BasicBlock = NUM_128;
        }
    }

    bool SetPseAlibiParamsRegbase() override
    {
        auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
        if (pseShape == nullptr) {
            return true;
        }
        if (pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            return true;
        }
        // 2: pre last axiss
        auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2);
        auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);

        PseEncodeType pseEncodeType = PseEncodeType::PSE_ENCODE_NONE;
        if (pseS1Size == PSE_ALIBI_S_SIZE && s1Size > PSE_ALIBI_S_SIZE) {
            if (s1Size == s2Size) {
                OP_CHECK_IF(inputParamsRegbase_->get_sparseType() != static_cast<uint8_t>(SparseEnum::CAUSAL),
                           OPS_REPORT_VECTOR_INNER_ERR(opName, "Pse alibi only support causal sparse type."), return false);
                pseEncodeType = PseEncodeType::PSE_ENCODE_ALIBI_S2_FULL;
            } else {
                OPS_REPORT_VECTOR_INNER_ERR(opName, "Pse alibi only support same S1 S2 when S1 lager than 1024");
                return false;
            }
        }
        inputParamsRegbase_->set_pseEncodeType(static_cast<uint8_t>(pseEncodeType));
        inputParamsRegbase_->set_pseS1Size(pseS1Size);
        inputParamsRegbase_->set_pseS2Size(pseS2Size);
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        uint8_t pseMode = hasPse ? static_cast<uint8_t>(pseType) : static_cast<uint8_t>(PseType::PSE_NONE_TYPE);
        OP_LOGD(opName, "TilingKey info is implMode:%d, s1TemplateType:%d, s2TemplateType:%d, dTemplateType:%d,"
            "dVTemplateType:%d, pseMode:%d, hasAttenMask:%d, hasDropOut:%d, hasRope:%d, outDtype:%d, regbase:%d",
            static_cast<uint8_t>(implMode), static_cast<uint16_t>(s1TemplateType), static_cast<uint16_t>(s2TemplateType),
            static_cast<uint16_t>(dTemplateType), static_cast<uint16_t>(dVTemplateType), pseMode, hasAttenMask,
            hasDropOut, hasRope, static_cast<uint8_t>(outDtype), static_cast<uint8_t>(regbase));

        // Const 128
        if (dTemplateType == dVTemplateType) {
            return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyLayout),
            static_cast<uint16_t>(s1TemplateType), static_cast<uint16_t>(s2TemplateType),
            static_cast<uint16_t>(dTemplateType), static_cast<uint16_t>(DTemplateType::NONALIGNED), pseMode, hasAttenMask,
            hasDropOut, hasRope, static_cast<uint8_t>(outDtype), static_cast<uint8_t>(regbase));
        }
        return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyLayout),
            static_cast<uint16_t>(s1TemplateType), static_cast<uint16_t>(s2TemplateType),
            static_cast<uint16_t>(dTemplateType), static_cast<uint16_t>(dVTemplateType), pseMode, hasAttenMask,
            hasDropOut, hasRope, static_cast<uint8_t>(outDtype), static_cast<uint8_t>(regbase));
    }

    bool IsCapable() override
    {
        if (socVersion != platform_ascendc::SocVersion::ASCEND910_95) {
            OP_LOGD(opName, "Current soc version is not platform_ascendc::SocVersion::ASCEND910_95.");
            return false;
        }

        return true;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        // 当前只有当D较大时需要开启workspace存放Bmm2的数据
        int64_t bmm2Bytes = 0;
        int64_t vec2Bytes = 0;
        int64_t bmm2ResBlockSize = dVBasicBlock;
        if (dTemplateType > DTemplateType::ALIGNED_256) {
            bmm2ResBlockSize = 512L;
        }
        bool useDn = (!hasPse && !hasAttenMask && !hasDropOut && s1BasicBlock != NUM_64
                      && dVBasicBlock <= NUM_256 && !hasRope);
        if ((!useDn && dSize > MIN_D_TO_USE_WORKSPACE) ||
            (useDn && dSize > DN_MIN_D_TO_USE_WORKSPACE)) {
            bmm2Bytes = s1BasicBlock * bmm2ResBlockSize * calcTypeSize;
            if (dTemplateType > DTemplateType::ALIGNED_256) {
                vec2Bytes = s1BasicBlock * dVBasicBlock * calcTypeSize;
            }
        }
        bmm2Bytes = AlignUp(bmm2Bytes, GM_ALIGN);
        vec2Bytes = AlignUp(vec2Bytes, GM_ALIGN);
        workspaces[0] = static_cast<size_t>((bmm2Bytes + vec2Bytes) * PING_PONG_VALUE *
            multiCoreParamsRegbase_->get_coreNum()) + WORK_SPACE_RESERVE_SIZE;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        FlashAttentionScoreConstTiling::PostTiling();
        return ge::GRAPH_SUCCESS;
    }
};

class FlashAttentionScoreTilingVarLenConst : public FlashAttentionScoreConstTiling {
public:
    explicit FlashAttentionScoreTilingVarLenConst(gert::TilingContext *context) :
        FlashAttentionScoreConstTiling(context)
    {
        this->templateName = "VarLenConst";
        this->regbase = true;
    }
    ~FlashAttentionScoreTilingVarLenConst() override = default;

protected:
    int64_t s2SizeLimitMax = 128;

    STemplateType s1TemplateType = STemplateType::STEMPLATEBOTTOM;
    STemplateType s2TemplateType = STemplateType::STEMPLATEBOTTOM;

    ge::graphStatus CheckContext() override {
        FlashAttentionScoreConstTiling::CheckContext();
        return ge::GRAPH_SUCCESS;
    }

    int64_t CalcTotalSize() override {
        int64_t totalSize = bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize();
        if (totalSize < static_cast<int64_t>(aicNum) && implMode != ImplMode::AA_INVALID_LINE_HIGH_PRECISION &&
            inputDtypeBytes != DATA_TYPE_FP32 && dBasicBlock <= NUM_256 && !hasRope) {
            if (s2Size > NUM_1024 && !hasAttenMask && !hasPse && !hasDropOut) {
                s2TemplateType = STemplateType::ALIGNED_256;
                s2BasicBlock = NUM_256;
            }
            s1TemplateType = STemplateType::ALIGNED_64;
            s1BasicBlock = NUM_64;
            multiCoreParamsRegbase_->set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
            totalSize = bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize();
        }
        return totalSize;
    }

    void CalcDBasicBlock() override {
        /* 先确定D的基本块，确定的逻辑是按照64来分档 */
        dBasicBlock = AlignUp(dSize + dSizeRope, D_TEMPLATE_SPLIT_SIZE);
        switch (dBasicBlock) {
            case NUM_64:
                dTemplateType = DTemplateType::ALIGNED_64;
                break;
            case NUM_128:
                dTemplateType = DTemplateType::ALIGNED_128;
                break;
            case NUM_192:
                dTemplateType = DTemplateType::ALIGNED_192;
                break;
            case NUM_256:
                dTemplateType = DTemplateType::ALIGNED_256;
                break;
            case NUM_320:
            case NUM_384:
            case NUM_448:
            case NUM_512:
                dTemplateType = DTemplateType::ALIGNED_512;
                break;
            default:
                dTemplateType = DTemplateType::DTEMPLATEBOTTOM;
        }
    }

    void CalcS1S2BasicBlock() override
    {
        /* 无可选输入使能dn优化 */
        if (dSize > NUM_256) {
            if (inputDtypeBytes == DATA_TYPE_FP32) {
                s1TemplateType = STemplateType::ALIGNED_64;
                s1BasicBlock = NUM_64;
            } else {
                s1TemplateType = STemplateType::ALIGNED_128;
                s1BasicBlock = NUM_128;
            }
            s2TemplateType = STemplateType::ALIGNED_128;
            s2BasicBlock = NUM_128;
        } else {
            s1TemplateType = STemplateType::ALIGNED_128;
            s2TemplateType = STemplateType::ALIGNED_128;
            s1BasicBlock = NUM_128;
            s2BasicBlock = NUM_128;
        }
    }

    bool SetBandLeftUpCausalPseParamsRegbase()
    {
        for (int64_t i = 0L; i < bSize; ++i) {
            if (i == 0) {
                if (actualSeqLenData[0] - actualSeqLenKvData[0] + qStartIdx - kvStartIdx == 0) {
                    continue;
                } else {
                    OP_LOGE(context_, "Inner pse sparse mode 8 is only supported when actualSeqLenData[0] %ld + qStartIdx %ld - actualSeqLenKvData[0] %ld - kvStartIdx %ld == 0.",
                    actualSeqLenData[0], qStartIdx, actualSeqLenKvData[0], kvStartIdx);
                    return false;
                }
            }
            if (actualSeqLenData[i] != actualSeqLenKvData[i]) {
                OP_LOGE(context_, "Inner pse sparse mode 8 is only supported when actualSeqQLen[%ld] %ld and actualSeqKvLen[%ld] %ld are equal.", i, actualSeqLenData[i], i, actualSeqLenKvData[i]);
                return false;
            }
        }
        return true;
    }

    bool SetPseAlibiParamsRegbase() override
    {
        auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
        if (pseShape == nullptr) {
            return true;
        }
        if (pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            OP_CHECK_IF(inputParamsRegbase_->get_sparseType() == static_cast<uint8_t>(SparseEnum::RIGHT_DOWN_CAUSAL_BAND),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "INNER Pse does not support sparse mode 7."), return false);
            if (inputParamsRegbase_->get_sparseType() == static_cast<uint8_t>(SparseEnum::BAND_LEFT_UP_CAUSAL)) {
                OP_CHECK_IF(!SetBandLeftUpCausalPseParamsRegbase(),
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "Inner pse sparse mode 8 param set error."), return false);
                return true;
            }
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] != actualSeqLenKvData[i]) {
                    OP_LOGE(context_, "Inner pse alibi is only support when actualSeqQLen and actualSeqKvLen are equal.");
                    return false;
                }
            }
            return true;
        }

        auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - DIM_NUM_2);
        auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);

        PseEncodeType pseEncodeType = PseEncodeType::PSE_ENCODE_NONE;
        OP_LOGD(context_, "[%s] pseS1Size:%ld, pseS2Size:%ld.", templateName, pseS1Size, pseS2Size);
        if (pseS1Size == PSE_ALIBI_S_SIZE) {
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] != actualSeqLenKvData[i]) {
                    OP_LOGE(context_, "Pse alibi only support when actualSeqQLen and actualSeqKvLen are equal.");
                    return false;
                }
            }
            OP_CHECK_IF(inputParamsRegbase_->get_sparseType() != static_cast<uint8_t>(SparseEnum::CAUSAL) &&
                           inputParamsRegbase_->get_sparseType() !=
                               static_cast<uint8_t>(SparseEnum::RIGHT_DOWN_CAUSAL),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "Pse alibi only support causal sparse type."), return false);
            pseEncodeType = PseEncodeType::PSE_ENCODE_ALIBI_S2_FULL;
            OP_LOGD(context_, "[%s] PSE_ENCODE_ALIBI_S2_FULL.", templateName);
        }
        inputParamsRegbase_->set_pseEncodeType(static_cast<uint8_t>(pseEncodeType));
        inputParamsRegbase_->set_pseS1Size(pseS1Size);
        inputParamsRegbase_->set_pseS2Size(pseS2Size);
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        uint8_t pseMode = hasPse ? static_cast<uint8_t>(pseType) : static_cast<uint8_t>(PseType::PSE_NONE_TYPE);
        OP_LOGD(opName, "TND TilingKey info is implMode:%d, s1TemplateType:%d, s2TemplateType:%d, dTemplateType:%d,"
            "pseMode:%d, hasAttenMask:%d, hasDropOut:%d, hasRope:%d, regbase:%d", static_cast<uint8_t>(implMode),
            static_cast<uint16_t>(s1TemplateType), static_cast<uint16_t>(s2TemplateType),
            static_cast<uint16_t>(dTemplateType), pseMode, hasAttenMask, hasDropOut, hasRope,
            static_cast<uint8_t>(regbase));

        // Const 128
        if (dTemplateType == dVTemplateType) {
            return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyLayout),
            static_cast<uint16_t>(s1TemplateType), static_cast<uint16_t>(s2TemplateType),
            static_cast<uint16_t>(dTemplateType), static_cast<uint16_t>(DTemplateType::NONALIGNED), pseMode, hasAttenMask,
            hasDropOut, hasRope, static_cast<uint8_t>(outDtype), static_cast<uint8_t>(regbase));
        }
        return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyLayout),
            static_cast<uint16_t>(s1TemplateType), static_cast<uint16_t>(s2TemplateType),
            static_cast<uint16_t>(dTemplateType), static_cast<uint16_t>(dVTemplateType), pseMode, hasAttenMask,
            hasDropOut, hasRope, 0, static_cast<uint8_t>(regbase));
    }

    bool IsCapable() override
    {
        if (socVersion != platform_ascendc::SocVersion::ASCEND910_95) {
            OP_LOGD(opName, "Current soc version is not platform_ascendc::SocVersion::ASCEND910_95.");
            return false;
        }
        if (tilingKeyLayout != LayoutType::LAYOUT_TND) {
            return false;
        }
        return true;
    }

    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum) override
    {
        (void)coreNum;
        int64_t accumS1BlockNum = 0;
        for (int64_t i = 0; i < bSize; ++i) {
            accumS1BlockNum += CeilDivision(actualSeqLenData[i], s1BasicBlock);
        }
        totalSize = accumS1BlockNum * n2Size * gSize;
        int64_t actualUsedCoreNum = std::min(totalSize, static_cast<int64_t>(aicNum));
        multiCoreParamsRegbase_->set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
        multiCoreParamsRegbase_->set_totalSize(totalSize);
        multiCoreParamsRegbase_->set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
        multiCoreParamsRegbase_->set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParamsRegbase_->get_splitFactorSize()));
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        // 当前只有当D较大时需要开启workspace存放Bmm2的数据
        int64_t bmm2Bytes = 0;
        int64_t vec2Bytes = 0;
        int64_t bmm2ResBlockSize = dBasicBlock;
        if (dTemplateType > DTemplateType::ALIGNED_256) {
            bmm2ResBlockSize = 512L;
        }
        if (dSize > MIN_D_TO_USE_WORKSPACE) {
            bmm2Bytes = s1BasicBlock * bmm2ResBlockSize * calcTypeSize;
            if (dTemplateType > DTemplateType::ALIGNED_256) {
                vec2Bytes = s1BasicBlock * dBasicBlock * calcTypeSize;
            }
        }
        bmm2Bytes = AlignUp(bmm2Bytes, GM_ALIGN);
        vec2Bytes = AlignUp(vec2Bytes, GM_ALIGN);
        workspaces[0] = static_cast<size_t>(((bmm2Bytes + vec2Bytes) * PING_PONG_VALUE) *
            multiCoreParamsRegbase_->get_coreNum()) + WORK_SPACE_RESERVE_SIZE;
        return ge::GRAPH_SUCCESS;
    }

    bool CheckBandPretokenAndNexttoken(SparseEnum &sparseType)
    {
        if (preTokens < 0) {
            OP_LOGE(context_, "pre_tokens[%ld] config error, has invalid data block.", preTokens);
            return false;
        }
        if (nextTokens < 0 && preTokens + nextTokens < 0) {
            OP_LOGE(context_, "pre_tokens[%ld], next_tokens[%ld], invalid config.", preTokens, nextTokens);
            return false;
        }
        for (int64_t i = 0L; i < bSize; ++i) {
            if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                continue;
            }
            if (actualSeqLenData[i] - nextTokens > actualSeqLenKvData[i]) {
                OP_LOGE(context_, "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row.", i,
                            actualSeqLenData[i], actualSeqLenKvData[i], nextTokens);
                return false;
            }
        }
        if (preTokens >= s2Size && nextTokens == 0) {
            preTokens = s2Size;
            nextTokens = 0;
            sparseType = SparseEnum::RIGHT_DOWN_CAUSAL;
        } else {
            sparseType = SparseEnum::BAND_COMPRESS;
        }
        return true;
    }

    bool CheckRightDownCausalBandPretokenAndNexttoken(SparseEnum &sparseType)
    {
        int64_t lastS2 = actualSeqLenKvData[bandIndex];
        if (preTokens < lastS2 || nextTokens > 0) {
            OP_LOGE(context_,
                        "RightDownCausal_Band mode: pre_tokens[%ld] is smaller than last valid s2[%ld]"
                        "or next_tokens[%ld] is larger than 0, wrong config.",
                        preTokens, lastS2, nextTokens);
            return false;
        }
        for (int64_t i = 0L; i < bSize; ++i) {
            if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                continue;
            }
            if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                OP_LOGE(context_, "Batch[%ld] s1[%ld] is larger than s2[%ld].", i, actualSeqLenData[i],
                            actualSeqLenKvData[i]);
                return false;
            }
            if ((i == bandIndex) && (actualSeqLenData[i] - nextTokens > actualSeqLenKvData[i])) {
                OP_LOGE(context_, "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row.", i,
                            actualSeqLenData[i], actualSeqLenKvData[i], nextTokens);
                return false;
            }
        }
        sparseType = SparseEnum::RIGHT_DOWN_CAUSAL_BAND;
        return true;
    }

    bool CheckPretokenAndNexttoken(SparseEnum &sparseType)
    {
        if (sparseMode == static_cast<int64_t>(SparseMode::ALL_MASK)) {
            if (preTokens < s1Size - 1 || nextTokens < s2Size - 1) {
                OP_LOGW(context_,
                          "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                          "preTokens and nextTokens will be reset max int value.",
                          preTokens, nextTokens, sparseMode);
                preTokens = std::numeric_limits<int32_t>::max();
                nextTokens = std::numeric_limits<int32_t>::max();
            }
            sparseType = static_cast<SparseEnum>(static_cast<uint8_t>(sparseMode));
        } else if (sparseMode == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) {
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
            sparseType = SparseEnum::CAUSAL;
        } else if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                    OP_LOGE(context_, "Batch[%ld] s1[%ld] is larger than s2[%ld], exist invalid row.", i,
                              actualSeqLenData[i], actualSeqLenKvData[i]);
                    return false;
                }
            }
            preTokens = s2Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
            sparseType = SparseEnum::RIGHT_DOWN_CAUSAL;
        } else if (sparseMode == static_cast<int64_t>(SparseMode::BAND)) {
            OP_CHECK_IF(!CheckBandPretokenAndNexttoken(sparseType),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "Check band mode pre_tokens and next_tokens fail."),
                       return false);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL_BAND)) {
            OP_CHECK_IF(!CheckRightDownCausalBandPretokenAndNexttoken(sparseType),
                       OPS_REPORT_VECTOR_INNER_ERR(opName,
                       "Check right down causal band mode pre_tokens and next_tokens fail."),
                       return false);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)) {
            if (actualSeqLenData[bandIndex] - nextTokens > actualSeqLenKvData[bandIndex]) {
                OP_LOGE(context_, "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row.", bandIndex,
                          actualSeqLenData[0], actualSeqLenKvData[0], nextTokens);
                return false;
            }
            int64_t firstS2 = actualSeqLenKvData[bandIndex];
            if (preTokens < firstS2) {
                OP_LOGE(context_, "Band_LeftUpCausal mode: pre_tokens[%ld] is smaller than first valid s2[%ld].",
                          preTokens, firstS2);
                return false;
            }
            sparseType = SparseEnum::BAND_LEFT_UP_CAUSAL;
        }
        return true;
    }

    bool SparseNoMaskModeCheck(int64_t maxS1Val, int64_t maxS2Val, int64_t minS2Val,
                               SparseEnum &sparseType)
    {
        if (nextTokens < 0) {
            OP_LOGE(context_, "nextTokens[%ld] config error, there is no valid data block.", nextTokens);
            return false;
        }
        if (preTokens >= maxS1Val && nextTokens >= maxS2Val) {
            return true;
        }
        for (int64_t i = 0L; i < bSize; ++i) {
            if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                continue;
            }
            if (actualSeqLenKvData[i] + preTokens < actualSeqLenData[i]) {
                OP_LOGE(context_, "Batch[%ld] s1[%ld] s2[%ld] has invalid row, check pre_tokens and next_tokens.", i,
                          actualSeqLenData[i], actualSeqLenKvData[i]);
                return false;
            }
        }
        if (preTokens >= 0) {
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            sparseType = SparseEnum::BAND;
            return true;
        }

        if (preTokens < 0) {
            int64_t absPreTokens = std::abs(preTokens);
            if (nextTokens >= absPreTokens) {
                s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), 0L);
                s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
                sparseType = SparseEnum::BAND;
                return true;
            } else {
                OP_LOGE(context_,
                          "preTokens[%ld] and nextTokens[%ld] config error with S[%ld], has invalid data block.",
                          preTokens, nextTokens, minS2Val);
                return false;
            }
        }
        return true;
    }

    bool VarLenGetPrefixNList(SparseEnum &sparseType)
    {
        auto prefixN = context_->GetOptionalInputTensor(PREFIX_INPUT_INDEX);
        if (prefixN == nullptr) {
            OP_LOGE(context_, "[%s] prefixN is null pointer while sparse mode is prefix compress", templateName);
            return false;
        }

        auto &prefixShape = prefixN->GetShape().GetStorageShape();
        if (prefixShape.GetDimNum() != 1) {
            OP_LOGE(context_, "[%s] prefixN shape is invalid, DimNum should be 1, but it is %lu.", templateName,
                      prefixShape.GetDimNum());
            return false;
        }
        if (prefixShape.GetDim(0) != bSize) {
            OP_LOGE(context_,
                      "[%s] prefixN is invalid, it should be the same size as bSize[%ld], but it "
                      "is %ld.",
                      templateName, bSize, prefixShape.GetDim(0));
            return false;
        }

        /* Get Data from tensor. */
        prefixNData = prefixN->GetData<int64_t>();
        if (prefixNData == nullptr) {
            OP_LOGE(context_, "[%s] prefixN data is null pointer", templateName);
            return false;
        }

        for (int64_t i = 0; i < bSize; ++i) {
            if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                if (prefixNData[i] < 0 || prefixNData[i] > actualSeqLenKvData[i]) {
                    OP_LOGE(context_, "[%s] batch[%ld] prefixN=%ld is invalid, should be in range of [0, %ld]",
                              templateName, i, prefixNData[i], actualSeqLenKvData[i]);
                    return false;
                }
                if (prefixNData[i] == 0) {
                    implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
                    OP_LOGD(context_, "Enable invalid line impl mode.");
                }
            } else {
                if (prefixNData[i] < actualSeqLenKvData[i] - actualSeqLenData[i] ||
                    prefixNData[i] > actualSeqLenKvData[i]) {
                    OP_LOGE(context_, "[%s] batch[%ld] prefixN=%ld is invalid, should be in range of [%ld, %ld]",
                              templateName, i, prefixNData[i], actualSeqLenKvData[i] - actualSeqLenData[i],
                              actualSeqLenKvData[i]);
                    return false;
                }
            }
        }

        sparseType = SparseEnum::PREFIX;
        return true;
    }

    bool VarLenSparseModeProcess(SparseEnum &sparseType)
    {
        if (!CheckPretokenAndNexttoken(sparseType)) {
            OP_LOGE(context_, "Check pre_tokens and next_tokens failed.");
            return false;
        }

        if (sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
            if (!VarLenGetPrefixNList(sparseType)) {
                return false;
            }
        }
        return true;
    }

    bool GetSparseInfo(SparseEnum &sparseType) override
    {
        OP_LOGD(context_,
                  "check sparse feature: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], attenMaskExist[%d]",
                  preTokens, nextTokens, s1Size, s2Size, hasAttenMask);
        if (sparseMode == static_cast<int64_t>(SparseMode::PREFIX) ||
            sparseMode > static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)) {
            OP_LOGE(context_, "Var len not support sparse mode %ld.", sparseMode);
            return false;
        }

        if (!hasAttenMask || tilingKeyLayout != LayoutType::LAYOUT_TND) {
            return true;
        }

        // if sparseMode is NoMask, preTokens and nextTokens start from top left vertex;
        // if sparseMode is Band, preTokens and nextTokens start from bottom right vertex.
        if (sparseMode == static_cast<int64_t>(SparseMode::NO_MASK)) {
            if (preTokens >= s1Size && nextTokens == 0) {
                sparseType = SparseEnum::CAUSAL;
                preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            } else {
                if (preTokens >= s1Size && nextTokens >= s2Size) {
                    return true;
                }
                int64_t minS2Val = *std::min_element(actualSeqLenKvData.begin(), actualSeqLenKvData.begin() + bSize);
                if (!SparseNoMaskModeCheck(s1Size, s2Size, minS2Val, sparseType)) {
                    return false;
                }
            }
        } else {
            if (!VarLenSparseModeProcess(sparseType)) {
                return false;
            }
        }
        return true;
    }

    int64_t GetS2RealSize(uint8_t sparseType, int32_t bOutIdx, int64_t s1OutIdx)
    {
        int64_t s2RealSize = s2Size;
        if (sparseType == static_cast<uint8_t>(SparseEnum::CAUSAL) && s1Size == s2Size) {
            s2RealSize = s1BasicBlock * (s1OutIdx + 1);
        } else if (sparseType == static_cast<uint8_t>(SparseEnum::PREFIX)) {
            s2RealSize = std::max(s1BasicBlock * (s1OutIdx + 1) - s1Size + s2Size, prefixNData[bOutIdx]);
        }
        return std::min(s2RealSize, actualSeqLenKvData[bOutIdx]);
    }

    bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx) override
    {
        (void)bIdx;
        OP_CHECK_IF(sparseValidArray.size() == 0,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "Sparse valid array size should be larger than 0."),
                   return false);

        // 特殊系数, 代表s2Size=[128, 256, 384, 512, 640, 768, 896, 1024] 对应的真实耗时膨胀值
        // 如 {256, 384, 512, 640, 768, 896, 960, 1024}; {384, 512, 640, 768, 832, 896, 960, 1024};
        // 和 {256, 256, 512, 512, 768, 768, 1024, 1024};
        // cof数组可变化，以下为当前选择参数，可优化不对齐场景负载均衡
        const int64_t cof[] = {128, 256, 384, 512, 640, 768, 896, 960, 1024};
        uint8_t sparseType = inputParamsRegbase_->get_sparseType();
        int64_t localAccumS1BlockNum = 0;
        for (int32_t i = 0; i < bSize; i++) {
            int64_t n2G = n2Size * gSize;
            int64_t s1BlockNum = CeilDivision(actualSeqLenData[i], s1BasicBlock);
            for (int64_t j = 0; j < s1BlockNum; j++) {
                // 此处暂时设置为1, 由于实测尾块1和128性能差距不大，理论上应该如下所示
                // 理论值: s1RealSize为std::min(s1BasicBlock, (actualSeqLenData[i] - s1BasicBlock * j))
                int64_t s1RealSize = 1;
                int64_t s2RealSize = GetS2RealSize(sparseType, i, j);
                // 新增一个系数, 解决理论和实际的差异
                int64_t s2RemainSize = s2RealSize % s2SizeLimitMax;
                s2RealSize = (s2RealSize / s2SizeLimitMax) * s2SizeLimitMax;
                int64_t s2SizeOffset = ((s2RemainSize > 0) ? cof[CeilDivision(s2RemainSize, 128L) - 1] : 0);
                s2RealSize += s2SizeOffset;

                // 每个s1方向上切分块的计算量
                for (int64_t k = 0; k < n2G; k++) {
                    sparseValidArray[localAccumS1BlockNum + k * s1BlockNum + j] = s1RealSize * s2RealSize;
                }
            }
            localAccumS1BlockNum += (s1BlockNum * n2G);
        }

        return true;
    }

    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, MultiCoreParamsRegbase &multiCoreParamsRegbase,
                     std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx)
    {
        // to avoid buffer overflow, or maybe sometimes we want to only verify single core
        int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParamsRegbase.get_coreNum()),
                                       static_cast<int64_t>(aicNum));
        int64_t totalSize = multiCoreParamsRegbase.get_totalSize();
        int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
        int64_t tmpMaxVal = maxVal;

        // 从前往后遍历
        for (int64_t idx = 1; idx < validAivNum; ++idx) {
            int64_t start = sparseStartIdx[idx];
            if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
                localValue[idx - 1] += sparseValidArray[start];
                localValue[idx] -= sparseValidArray[start];
                sparseStartIdx[idx] += 1;
            } else if (start == totalSize) {
                break;
            }
        }
        tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

        // 从后往前遍历
        for (int64_t idx = validAivNum - 1; idx > 0; --idx) {
            int64_t start = sparseStartIdx[idx];
            if (start == totalSize) {
                if (sparseStartIdx[idx - 1] == totalSize) {
                    continue;
                }
                localValue[idx - 1] -= sparseValidArray[start - 1];
                localValue[idx] = sparseValidArray[start - 1];
                sparseStartIdx[idx] -= 1;
            } else if (start > 0) {
                if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                    continue;
                }
                localValue[idx - 1] -= sparseValidArray[start - 1];
                localValue[idx] += sparseValidArray[start - 1];
                sparseStartIdx[idx] -= 1;
            } else {
                break;
            }
        }
        tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

        return (tmpMaxVal >= maxVal) ? false : true;
    }

    inline bool InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAivNum, int64_t totalSize,
                              const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue)
    {
        for (int64_t idx = 0; idx < validAivNum; ++idx) {
            int64_t start = sparseStartIdx[idx];
            int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
            if (start < totalSize) {
                localValue[idx] =
                    std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0LL);
            } else {
                break;
            }
        }
        return true;
    }

    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, MultiCoreParamsRegbase &multiCoreParamsRegbase,
                           int64_t maxCoreNum) override
    {
        (void)maxCoreNum;
        // to avoid buffer overflow, or maybe sometimes we want to only verify single core
        int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParamsRegbase.get_coreNum()),
                                       static_cast<int64_t>(aicNum));
        int64_t totalSize = multiCoreParamsRegbase.get_totalSize(); // BN2GS1.o
        int64_t *sparseStartIdx = multiCoreParamsRegbase.get_sparseStartIdxPtr();

        OP_CHECK_IF(totalSize <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "totalSize should be larger than 0."),
                   return false);

        // initLoad: 使用均分策略, 保证后续不会比均分差
        int64_t splitFactorSize = multiCoreParamsRegbase.get_splitFactorSize();
        std::vector<int64_t> localSparseStartIdx(aicNum, totalSize);
        for (int64_t idx = 0; idx < static_cast<int64_t>(aicNum); ++idx) {
            localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
        }
        std::vector<int64_t> localValue(validAivNum, 0);
        InitLoadValue(sparseValidArray, validAivNum, totalSize, localSparseStartIdx, localValue);

        // 负载均衡粗调
        std::vector<int64_t> tmpLocalValue(validAivNum, 0);
        std::vector<int64_t> tmpsparseStartIdx(aicNum, totalSize);
        int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
        int64_t avgVal = CeilDivision(sparseArraySum, validAivNum);

        tmpsparseStartIdx[0] = 0;
        for (int64_t idx = 1; idx < static_cast<int64_t>(aicNum); ++idx) {
            int64_t start = tmpsparseStartIdx[idx - 1];
            int64_t singleLoadValue = 0;
            tmpsparseStartIdx[idx] = start;
            while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
                singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
                tmpsparseStartIdx[idx] += 1;
            }

            if ((start + 1) < tmpsparseStartIdx[idx]) {
                int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
                tmpsparseStartIdx[idx] = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ?
                                         (tmpsparseStartIdx[idx] - 1) : (tmpsparseStartIdx[idx]);
                singleLoadValue = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ? redoSingleLoadValue :
                                                                                                  singleLoadValue;
                sparseArraySum -= singleLoadValue;
                avgVal = CeilDivision(sparseArraySum, (validAivNum - idx));
            }
        }

        InitLoadValue(sparseValidArray, validAivNum, totalSize, tmpsparseStartIdx, tmpLocalValue);

        // 负载均衡精调
        while (BalanceLoad(sparseValidArray, multiCoreParamsRegbase, tmpLocalValue, tmpsparseStartIdx)) {
            // 根据负载均衡是否能得到更好预测结果决定是否结束循环
        }

        // exchange initLoad and 负载均衡
        if ((*std::max_element(localValue.begin(), localValue.end())) >
            (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
            localSparseStartIdx.swap(tmpsparseStartIdx);
            localValue.swap(tmpLocalValue);
        }
        for (int64_t idx = 0; idx < static_cast<int64_t>(aicNum); ++idx) {
            sparseStartIdx[idx] = localSparseStartIdx[idx];
        }

        return true;
    }

    void SetSparseParamsRegbase(int64_t maxCoreNum) override
    {
        std::vector<int64_t> sparseValidArray(multiCoreParamsRegbase_->get_totalSize(), 0);
        InitSparseValidArray(sparseValidArray, 0);
        SetSparseStartIdx(sparseValidArray, *multiCoreParamsRegbase_, maxCoreNum);

        inputParamsRegbase_->set_s1Size(realT1Size);
        inputParamsRegbase_->set_s1SparseValidSize(s1SparseValidSize);
        inputParamsRegbase_->set_s2SparseValidSize(s2SparseValidSize);
    }

    ge::graphStatus PostTiling() override
    {
        FlashAttentionScoreConstTiling::PostTiling();
        return ge::GRAPH_SUCCESS;
    }
};

class FlashAttentionScoreTilingDropMaskRegbase : public FlashAttentionScoreConstTiling {
public:
    explicit FlashAttentionScoreTilingDropMaskRegbase(gert::TilingContext *context) :
        FlashAttentionScoreConstTiling(context)
    {
        // manually initialize tiling data in hostapi scenario
        OP_CHECK_IF(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                            context_->GetRawTilingData()->GetCapacity()) != EOK,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "Fail to memset tiling data"), return;);
    }
    ~FlashAttentionScoreTilingDropMaskRegbase() override = default;

protected:
    ge::graphStatus DoOpTiling() override
    {
        if (!inputParamsRegbase_->get_needDropMaskOp()) {
            return ge::GRAPH_PARAM_INVALID;
        }

        int64_t shapeTotalSize = inputParamsRegbase_->get_bSize() * inputParamsRegbase_->get_n2Size() *
                                 inputParamsRegbase_->get_gSize() * inputParamsRegbase_->get_s1Size() *
                                 inputParamsRegbase_->get_s2Size();
        auto layoutType = inputParamsRegbase_->get_layoutType();
        if (layoutType == static_cast<uint8_t>(LayoutType::LAYOUT_TND)) {
            for (auto i = 0; i < bSize; i++) {
                dropTotalSize += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeTotalSize = inputParamsRegbase_->get_n2Size() * inputParamsRegbase_->get_gSize() * dropTotalSize;
            OP_LOGD(context_, "shapeTotalSize %ld dropTotalSize %ld.", shapeTotalSize, dropTotalSize);
        }
        // 保证每核计算数据量256倍数，2048表示bit位，256 * 8
        const int64_t ubCalFactor = 2048;
        int64_t shapeSplitCoreSize = CeilDivision(shapeTotalSize, ubCalFactor);
        int64_t shapeSingleCoreSize = CeilDivision(shapeSplitCoreSize, static_cast<int64_t>(aivNum));

        // ub能计算的最大元素数, 向下对齐
        // 单次ub计算量为x个元素,空间占用：x/8 * 1 [1个 uint8]+ 2x * 2[2个fp16,select的src和res] + x * 1 [1个uint8]共6份
        int64_t baseUbCalSize = AlignDown(CeilDivision(static_cast<int64_t>(aicoreParams_.ubSize),
                                          DROP_MASK_CAL_SIZE), ubCalFactor);
        baseUbCalSize = std::min(baseUbCalSize, shapeSingleCoreSize * ubCalFactor);
        // ub 的外层循环次数
        int64_t multiCoreFactorSize = CeilDivision(shapeSingleCoreSize * ubCalFactor, baseUbCalSize);
        shapeTotalSize = AlignUp(shapeTotalSize, GM_ALIGN);

        dropmaskParamsRegbase_->set_shapeTotalSize(shapeTotalSize);
        dropmaskParamsRegbase_->set_multiCoreFactorSize(static_cast<int32_t>(multiCoreFactorSize));
        dropmaskParamsRegbase_->set_multiCoreTotalSize(CeilDivision(shapeSplitCoreSize * ubCalFactor, baseUbCalSize));
        dropmaskParamsRegbase_->set_baseUbCalSize(static_cast<int32_t>(baseUbCalSize));
        return ge::GRAPH_PARAM_INVALID;
    }

    uint64_t GetTilingKey() const override
    {
        return 0UL;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    void CalcS1S2BasicBlock() override {}

    void CalcDBasicBlock() override {}
};

REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(FlashAttentionScore, FlashAttentionScoreTilingDropMaskRegbase, (int32_t)platform_ascendc::SocVersion::ASCEND910_95, 81);
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(FlashAttentionScore, FlashAttentionScoreTilingVarLenConst, (int32_t)platform_ascendc::SocVersion::ASCEND910_95, 82);
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(FlashAttentionScore, FlashAttentionScoreTilingS1S2Const, (int32_t)platform_ascendc::SocVersion::ASCEND910_95, 83);
} // namespace FA
} // namespace optiling
