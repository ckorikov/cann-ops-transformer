#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry_base.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"
#include "common/utils/ut_op_common.h"
#include "experiment_ops.h"
#include "fusion_ops.h"
#include "array_ops.h"
#include "error_util.h"
#include "util/util.h"

namespace
{
static const size_t INDEX_OUT_GMM_Y = 0;
static const size_t INDEX_OUT_MM_Y = 1;
static const size_t INDEX_PERMUTE_OUT = 2;

static constexpr size_t DIM_NUM_1 = 1;
static constexpr size_t DIM_NUM_2 = 2;
static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;

struct TestParam {
    string test_name{};
    std::vector<std::pair<string, string>> tiling_params_str_pair{};
    std::vector<std::pair<string, std::vector<int64_t>>> tiling_params_vec_pair{};
    std::vector<std::pair<size_t, ge::DataType>> tiling_input_dtypes_pair{};
    std::vector<std::pair<size_t, ge::DataType>> tiling_output_dtypes_pair{};
    ge::graphStatus status;
};

struct TilingParams {
    uint64_t BSK{4096};
    uint64_t BS{2048};
    uint64_t K{2};
    uint64_t H1{7168};
    uint64_t H2{7168};
    uint64_t A{4096};
    uint64_t N1{4096};
    uint64_t N2{4096};
    uint64_t ep_world_size{8};
    uint64_t e{4};
    uint64_t aivCoreNum{40};
    uint64_t aicCoreNum{20};
    uint64_t gmm_weight_dim1{7168};
    uint64_t y_dim1{4096};
    uint64_t mm_weight_dim0{7168};
    bool trans_gmm_weight{false};
    bool trans_mm_weight{false};
    bool permute_out_flag{false};
    std::string group{"group"};
    std::vector<int64_t> send_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recv_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

struct TilingShapes {
    gert::StorageShape gmm_x_shape;
    gert::StorageShape gmm_weight_shape;
    gert::StorageShape send_counts_shape;
    gert::StorageShape recv_counts_shape;
    gert::StorageShape mm_x_shape;
    gert::StorageShape mm_weight_shape;

    gert::StorageShape y_shape;
    gert::StorageShape mm_y_shape;
    gert::StorageShape permute_out_shape;
};

struct TilingDTypes {
    std::vector<ge::DataType> input_dtypes{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT64,
                                           ge::DT_INT64,   ge::DT_FLOAT16, ge::DT_FLOAT16};
    std::vector<ge::DataType> output_dtypes{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
};

class AlltoAllvGroupedMatMulRuntimeProtoTest : public testing::TestWithParam<TestParam>
{
protected:
    static void SetUpTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
        std::cout << "AlltoAllvGroupedMatMulRuntimeProtoTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulRuntimeProtoTest TearDown" << std::endl;
    }

    void InitTilingParams(const TestParam& test_param)
    {
        this->tiling_params = TilingParams{};
        this->InitTilingStrParams(test_param.tiling_params_str_pair);
        this->InitTilingVecParams(test_param.tiling_params_vec_pair);
    }

    void InitTilingStrParams(const std::vector<std::pair<string, string>>& tiling_params_pair)
    {
        auto& tiling_params = this->tiling_params;
        auto& tiling_params_str_handlers = AlltoAllvGroupedMatMulRuntimeProtoTest::tiling_params_str_handlers;
        for (auto& kv : tiling_params_pair) {
            if (tiling_params_str_handlers.count(kv.first) != 0) {
                tiling_params_str_handlers[kv.first](tiling_params, kv.second);
            }
        }
    }

    void InitTilingVecParams(const std::vector<std::pair<string, std::vector<int64_t>>>& tiling_params_pair)
    {
        auto& tiling_params = this->tiling_params;
        auto& tiling_params_vec_handlers = AlltoAllvGroupedMatMulRuntimeProtoTest::tiling_params_vec_handlers;
        for (auto& kv : tiling_params_pair) {
            if (tiling_params_vec_handlers.count(kv.first) != 0) {
                tiling_params_vec_handlers[kv.first](tiling_params, kv.second);
            }
        }
    }

    void InitTilingShape()
    {
        auto& tiling_params = this->tiling_params;
        auto& BSK = tiling_params.BSK;
        auto& BS = tiling_params.BS;
        auto& K = tiling_params.K;
        auto& H1 = tiling_params.H1;
        auto& H2 = tiling_params.H2;
        auto& A = tiling_params.A;
        auto& N1 = tiling_params.N1;
        auto& N2 = tiling_params.N2;
        auto& ep_world_size = tiling_params.ep_world_size;
        auto& e = tiling_params.e;
        auto& gmm_weight_dim1 = tiling_params.gmm_weight_dim1;
        auto& y_dim1 = tiling_params.y_dim1;
        auto& mm_weight_dim0 = tiling_params.mm_weight_dim0;

        auto& tiling_shapes = this->tiling_shapes;
        tiling_shapes.gmm_x_shape = {{BSK, H1}, {BSK, H1}};
        tiling_shapes.gmm_weight_shape = {{e, gmm_weight_dim1, N1}, {e, gmm_weight_dim1, N1}};
        tiling_shapes.send_counts_shape = {{e * ep_world_size}, {e * ep_world_size}};
        tiling_shapes.recv_counts_shape = {{e * ep_world_size}, {e * ep_world_size}};
        tiling_shapes.mm_x_shape = {{BS, H2}, {BS, H2}};
        tiling_shapes.mm_weight_shape = {{mm_weight_dim0, N2}, {mm_weight_dim0, N2}};

        tiling_shapes.y_shape = {{}, {}};
        tiling_shapes.mm_y_shape = {{}, {}};
        tiling_shapes.permute_out_shape = {{}, {}};
    }

    void InitTilingDTypes(const std::vector<std::pair<size_t, ge::DataType>>& tiling_input_dtypes_pair,
                          const std::vector<std::pair<size_t, ge::DataType>>& tiling_output_dtypes_pair)
    {
        auto& tiling_dtypes = this->tiling_dtypes;
        tiling_dtypes = TilingDTypes{};
        InitTilingInputDTypes(tiling_input_dtypes_pair);
        InitTilingOutputDTypes(tiling_output_dtypes_pair);
    }

    void InitTilingInputDTypes(const std::vector<std::pair<size_t, ge::DataType>>& tiling_input_dtypes_pair)
    {
        auto& tiling_dtypes = this->tiling_dtypes;
        for (auto& kv : tiling_input_dtypes_pair) {
            if (kv.first >= 0 && kv.first < tiling_dtypes.input_dtypes.size()) {
                tiling_dtypes.input_dtypes[kv.first] = kv.second;
            }
        }
    }

    void InitTilingOutputDTypes(const std::vector<std::pair<size_t, ge::DataType>>& tiling_output_dtypes_pair)
    {
        auto& tiling_dtypes = this->tiling_dtypes;
        for (auto& kv : tiling_output_dtypes_pair) {
            if (kv.first >= 0 && kv.first < tiling_dtypes.output_dtypes.size()) {
                tiling_dtypes.output_dtypes[kv.first] = kv.second;
            }
        }
    }

    void InitHolder(const TestParam& test_param)
    {
        this->InitTilingParams(test_param);
        auto& tiling_params = this->tiling_params;
        auto& group = tiling_params.group;
        auto& ep_world_size = tiling_params.ep_world_size;
        auto& send_counts = tiling_params.send_counts;
        auto& recv_counts = tiling_params.recv_counts;
        auto& trans_gmm_weight = tiling_params.trans_gmm_weight;
        auto& trans_mm_weight = tiling_params.trans_mm_weight;
        auto& permute_out_flag = tiling_params.permute_out_flag;

        this->InitTilingShape();
        auto& tiling_shapes = this->tiling_shapes;
        auto& gmm_x_shape = tiling_shapes.gmm_x_shape;
        auto& gmm_weight_shape = tiling_shapes.gmm_weight_shape;
        auto& send_counts_shape = tiling_shapes.send_counts_shape;
        auto& recv_counts_shape = tiling_shapes.recv_counts_shape;
        auto& mm_x_shape = tiling_shapes.mm_x_shape;
        auto& mm_weight_shape = tiling_shapes.mm_weight_shape;

        auto& y_shape = tiling_shapes.y_shape;
        auto& mm_y_shape = tiling_shapes.mm_y_shape;
        auto& permute_out_shape = tiling_shapes.permute_out_shape;

        this->InitTilingDTypes(test_param.tiling_input_dtypes_pair, test_param.tiling_output_dtypes_pair);
        auto& tiling_dtypes = this->tiling_dtypes;

        auto input_num = this->input_num;
        auto output_num = this->output_num;

        this->infer_shape_faker =
            gert::InferShapeContextFaker()
                .NodeIoNum(input_num, output_num)
                .IrInstanceNum({1, 1, 1, 1, 1, 1})
                .InputShapes({&gmm_x_shape, &gmm_weight_shape, nullptr, nullptr, &mm_x_shape, &mm_weight_shape})
                .OutputShapes({&y_shape, &mm_y_shape, &permute_out_shape})
                .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                            {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(ep_world_size)},
                            {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
                            {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
                            {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(trans_gmm_weight)},
                            {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(trans_mm_weight)},
                            {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(permute_out_flag)}});
        this->infer_shape_holder = this->infer_shape_faker.Build();

        std::vector<void*> input_dtypes_ptrs(input_num);
        for (int64_t i = 0; i < input_num; i++) {
            input_dtypes_ptrs[i] = &tiling_dtypes.input_dtypes[i];
        }
        std::vector<void*> output_dtypes_ptrs(output_num);
        this->infer_data_type_faker =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(input_num, output_num)
                .IrInstanceNum({1, 1, 1, 1, 1, 1})
                .InputDataTypes(input_dtypes_ptrs)
                .OutputDataTypes(output_dtypes_ptrs)
                .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                            {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(ep_world_size)},
                            {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
                            {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
                            {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(trans_gmm_weight)},
                            {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(trans_mm_weight)},
                            {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(permute_out_flag)}});
        this->infer_data_type_holder = this->infer_data_type_faker.Build();
    }

public:
    int64_t input_num{6};
    int64_t output_num{3};
    TilingParams tiling_params;
    TilingShapes tiling_shapes;
    TilingDTypes tiling_dtypes;
    gert::InferShapeContextFaker infer_shape_faker{};
    gert::KernelRunContextHolder infer_shape_holder{};
    gert::InferDataTypeContextFaker infer_data_type_faker{};
    gert::KernelRunContextHolder infer_data_type_holder{};
    static std::unordered_map<string, std::function<void(TilingParams& tiling_params, const string& value_str)>>
        tiling_params_str_handlers;
    static std::unordered_map<string,
                              std::function<void(TilingParams& tiling_params, const std::vector<int64_t> value_vec)>>
        tiling_params_vec_handlers;
};

std::unordered_map<string, std::function<void(TilingParams& tiling_params, const string& value_str)>>
    AlltoAllvGroupedMatMulRuntimeProtoTest::tiling_params_str_handlers = {
        {"BSK", [](TilingParams& tiling_params, const string& value_str) { tiling_params.BSK = std::stoi(value_str); }},
        {"BS", [](TilingParams& tiling_params, const string& value_str) { tiling_params.BS = std::stoi(value_str); }},
        {"K", [](TilingParams& tiling_params, const string& value_str) { tiling_params.K = std::stoi(value_str); }},
        {"H1", [](TilingParams& tiling_params, const string& value_str) { tiling_params.H1 = std::stoi(value_str); }},
        {"H2", [](TilingParams& tiling_params, const string& value_str) { tiling_params.H2 = std::stoi(value_str); }},
        {"A", [](TilingParams& tiling_params, const string& value_str) { tiling_params.A = std::stoi(value_str); }},
        {"N1", [](TilingParams& tiling_params, const string& value_str) { tiling_params.N1 = std::stoi(value_str); }},
        {"N2", [](TilingParams& tiling_params, const string& value_str) { tiling_params.N2 = std::stoi(value_str); }},
        {"ep_world_size", [](TilingParams& tiling_params,
                             const string& value_str) { tiling_params.ep_world_size = std::stoi(value_str); }},
        {"e", [](TilingParams& tiling_params, const string& value_str) { tiling_params.e = std::stoi(value_str); }},
        {"gmm_weight_dim1", [](TilingParams& tiling_params,
                               const string& value_str) { tiling_params.gmm_weight_dim1 = std::stoi(value_str); }},
        {"y_dim1",
         [](TilingParams& tiling_params, const string& value_str) { tiling_params.y_dim1 = std::stoi(value_str); }},
        {"mm_weight_dim0", [](TilingParams& tiling_params,
                              const string& value_str) { tiling_params.mm_weight_dim0 = std::stoi(value_str); }},
        {"trans_gmm_weight", [](TilingParams& tiling_params,
                                const string& value_str) { tiling_params.trans_gmm_weight = value_str == "true"; }},
        {"trans_mm_weight", [](TilingParams& tiling_params,
                               const string& value_str) { tiling_params.trans_mm_weight = value_str == "true"; }},
        {"permute_out_flag", [](TilingParams& tiling_params, const string& value_str) {
             tiling_params.permute_out_flag = value_str == "true";
         }}};

std::unordered_map<string, std::function<void(TilingParams& tiling_params, const std::vector<int64_t> value_vec)>>
    AlltoAllvGroupedMatMulRuntimeProtoTest::tiling_params_vec_handlers = {
        {"send_counts", [](TilingParams& tiling_params,
                           const std::vector<int64_t> value_vec) { tiling_params.send_counts = value_vec; }},
        {"recv_counts", [](TilingParams& tiling_params, const std::vector<int64_t> value_vec) {
             tiling_params.recv_counts = value_vec;
         }}};

TEST_P(AlltoAllvGroupedMatMulRuntimeProtoTest, runtime_proto_test)
{
    auto test_param = GetParam();
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);

    InitHolder(test_param);

    auto& infer_shape_holder = this->infer_shape_holder;
    auto& infer_data_type_holder = this->infer_data_type_holder;

    auto infer_shape_context = infer_shape_holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(infer_shape_context, nullptr);
    auto infer_data_type_context = infer_data_type_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(infer_data_type_context, nullptr);

    ASSERT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(infer_datatype_func(infer_data_type_context), ge::GRAPH_SUCCESS);

    auto* gmmYShape = infer_shape_context->GetOutputShape(INDEX_OUT_GMM_Y);
    EXPECT_EQ(gmmYShape->GetDimNum(), DIM_NUM_2);
    EXPECT_EQ(gmmYShape->GetDim(DIM_0), this->tiling_params.BSK);
    EXPECT_EQ(gmmYShape->GetDim(DIM_1), this->tiling_params.N1);
    auto gmmYDType = infer_data_type_context->GetOutputDataType(INDEX_OUT_GMM_Y);
    EXPECT_EQ(gmmYDType, ge::DT_FLOAT16);

    auto* mmYShape = infer_shape_context->GetOutputShape(INDEX_OUT_MM_Y);
    EXPECT_EQ(mmYShape->GetDimNum(), DIM_NUM_2);
    EXPECT_EQ(mmYShape->GetDim(DIM_0), this->tiling_params.BS);
    EXPECT_EQ(mmYShape->GetDim(DIM_1), this->tiling_params.N2);
    auto mmYDType = infer_data_type_context->GetOutputDataType(INDEX_OUT_MM_Y);
    EXPECT_EQ(mmYDType, ge::DT_FLOAT16);

    if (this->tiling_params.permute_out_flag) {
        auto* permute_out_shape = infer_shape_context->GetOutputShape(INDEX_PERMUTE_OUT);
        EXPECT_EQ(permute_out_shape->GetDimNum(), DIM_NUM_2);
        EXPECT_EQ(permute_out_shape->GetDim(DIM_0), this->tiling_params.A);
        EXPECT_EQ(permute_out_shape->GetDim(DIM_1), this->tiling_params.H1);
        auto permuteOutDType = infer_data_type_context->GetOutputDataType(INDEX_PERMUTE_OUT);
        EXPECT_EQ(permuteOutDType, ge::DT_FLOAT16);
    }
}

static TestParam test_params[] = {{"Test_sample", {{"permute_out_flag", "true"}}, {}, {}, {}, ge::GRAPH_SUCCESS}};

INSTANTIATE_TEST_SUITE_P(AlltoAllvGroupedMatMulRuntimeProtoTest, AlltoAllvGroupedMatMulRuntimeProtoTest,
                         testing::ValuesIn(test_params),
                         [](const testing::TestParamInfo<AlltoAllvGroupedMatMulRuntimeProtoTest::ParamType>& info) {
                             return info.param.test_name;
                         });
}  // namespace
