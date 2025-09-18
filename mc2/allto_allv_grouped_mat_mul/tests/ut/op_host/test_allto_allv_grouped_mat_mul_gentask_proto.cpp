#include "gtest/gtest.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "graph/gnode.h"
#include "graph/any_value.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/args_format_desc_utils.h"
#include "register/op_ext_gentask_registry.h"
#include "register/op_impl_space_registry.h"
#include "register/op_ct_impl_registry.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "error_util.h"
#define private public
#define protected public
#include "platform/platform_info.h"

namespace
{
struct TilingParams {
    uint64_t BSK{4096};
    uint64_t BS{2048};
    uint64_t K{2};
    uint64_t H1{7168};
    uint64_t H2{7168};
    uint64_t A{4096};
    uint64_t N1{4096};
    uint64_t N2{64};
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

class AlltoAllvGroupedMatMulGentaskProtoTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
        std::cout << "AlltoAllvGroupedMatMulGentaskProtoTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulGentaskProtoTest TearDown" << std::endl;
    }

    void InitTilingParams()
    {
        this->tiling_params = TilingParams{};
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

        tiling_shapes.y_shape = {{A, y_dim1}, {A, y_dim1}};
        tiling_shapes.mm_y_shape = {{BS, N2}, {BS, N2}};
        tiling_shapes.permute_out_shape = {{A, H1}, {A, H1}};
    }

    void InitTilingDTypes()
    {
        auto& tiling_dtypes = this->tiling_dtypes;
        tiling_dtypes = TilingDTypes{};
    }

    void InitHolder(const ge::Node& node, const std::string& opType)
    {
        this->InitTilingParams();
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

        this->InitTilingDTypes();
        auto& tiling_dtypes = this->tiling_dtypes;

        auto input_num = this->input_num;
        auto output_num = this->output_num;

        this->faker = gert::ExeResGenerationContextFaker()
                          .NodeIoNum(input_num, output_num)
                          .SetOpType(opType)
                          .IrInstanceNum({1, 1, 1, 1, 1})
                          .InputShapes({reinterpret_cast<void*>(const_cast<ge::Node*>(&node)), &gmm_x_shape,
                                        &gmm_weight_shape, nullptr, nullptr, &mm_x_shape, &mm_weight_shape})
                          .OutputShapes({&y_shape, &mm_y_shape, &permute_out_shape});
        this->holder = this->faker.Build();
    }

public:
    int64_t input_num{6};
    int64_t output_num{3};
    TilingParams tiling_params;
    TilingShapes tiling_shapes;
    TilingDTypes tiling_dtypes;
    gert::ExeResGenerationContextFaker faker{};
    gert::KernelRunContextHolder holder{};
};

static void SetSpaceRegistry(const char* opType)
{
    auto spaceRegistry = std::make_shared<gert::OpImplSpaceRegistry>();
    auto registryHolder = std::make_shared<gert::OpImplRegistryHolder>();
    auto funcs = gert::OpCtImplRegistry::GetInstance().GetOpImpl(opType);
    auto& ctImpl = registryHolder->GetTypesToCtImpl();
    ctImpl[opType] = *funcs;
    spaceRegistry->AddRegistry(registryHolder);
    gert::DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(spaceRegistry);
}

static void GenerateOpExtTask(const ge::Node& node, gert::KernelRunContextHolder& holder,
                              std::vector<domi::TaskDef>& tasks)
{
    const auto spaceRegistry =
        gert::DefaultOpImplSpaceRegistry::GetInstance().GetDefaultSpaceRegistry(node.GetOpDesc()->GetOppImplVersion());
    EXPECT_NE(spaceRegistry, nullptr);
    OP_LOGW("ut", "node.GetType() %s.", node.GetType().c_str());

    auto funcsPtr = spaceRegistry->GetOpCtImpl(node.GetType());
    EXPECT_NE(funcsPtr, nullptr);
    EXPECT_NE(funcsPtr->gen_task, nullptr);
    EXPECT_NE(funcsPtr->calc_op_param, nullptr);

    auto opExeResCtx = holder.GetContext<gert::ExeResGenerationContext>();
    EXPECT_NE(opExeResCtx, nullptr);

    std::vector<std::vector<uint8_t>> serializeTask;
    for (const auto& task : tasks) {
        size_t pbSize = task.ByteSizeLong();
        EXPECT_NE(pbSize, 0);
        std::vector<uint8_t> pbBuff(pbSize, 0);
        auto ret = task.SerializeToArray(pbBuff.data(), pbSize);
        EXPECT_EQ(ret, true);
        serializeTask.emplace_back(std::move(pbBuff));
    }

    auto ret = funcsPtr->calc_op_param(opExeResCtx);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    ret = funcsPtr->gen_task(opExeResCtx, serializeTask);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    tasks.clear();
    tasks.reserve(serializeTask.size());
    for (const auto& serialize_task : serializeTask) {
        domi::TaskDef task;
        EXPECT_EQ(task.ParseFromArray(serialize_task.data(), serialize_task.size()), true);
        tasks.emplace_back(std::move(task));
    }
}

TEST_F(AlltoAllvGroupedMatMulGentaskProtoTest, gen_task)
{
    // set version
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    platformInfo.str_info.short_soc_version = "Ascend910_93";
    optiCompilationInfo.soc_version = "Ascend910_93";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("AlltoAllvGroupedMatMulGraph");
    ge::OpDescPtr node_op = std::make_shared<ge::OpDesc>("AlltoAllvGroupedMatMulOpDesc", "AlltoAllvGroupedMatMul");
    ge::NodePtr node = graph->AddNode(node_op);
    node_op->SetId(0U);
    node_op->AppendIrInput("gmm_x", ge::IrInputType::kIrInputRequired);
    node_op->AppendIrInput("gmm_weight", ge::IrInputType::kIrInputRequired);
    node_op->AppendIrInput("send_counts_tensor", ge::IrInputType::kIrInputOptional);
    node_op->AppendIrInput("recv_counts_tensor", ge::IrInputType::kIrInputOptional);
    node_op->AppendIrInput("mm_x", ge::IrInputType::kIrInputOptional);
    node_op->AppendIrInput("mm_weight", ge::IrInputType::kIrInputOptional);
    node_op->AppendIrOutput("y", ge::IrOutputType::kIrOutputRequired);
    node_op->AppendIrOutput("mm_y", ge::IrOutputType::kIrOutputRequired);
    node_op->AppendIrOutput("permute_out", ge::IrOutputType::kIrOutputRequired);
    domi::TaskDef aicore_task{};
    aicore_task.set_type(RT_MODEL_TASK_FFTS_PLUS_TASK);
    domi::FftsPlusTaskDef* ffts_plus_task_def = aicore_task.mutable_ffts_plus_task();
    ffts_plus_task_def->set_op_index(0U);
    domi::FftsPlusCtxDef* ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    auto mixl2_task_ctx = ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx();
    std::vector<ge::ArgDesc> args_desc;
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::FFTS_ADDR, 0);
    mixl2_task_ctx->add_task_addr(1);
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::INPUT, 0);
    mixl2_task_ctx->add_task_addr(2);
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::INPUT, 1);
    mixl2_task_ctx->add_task_addr(3);
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::INPUT, 2);
    mixl2_task_ctx->add_task_addr(4);
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::INPUT, 3);
    mixl2_task_ctx->add_task_addr(5);
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::INPUT, 4);
    mixl2_task_ctx->add_task_addr(6);
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::INPUT, 5);
    mixl2_task_ctx->add_task_addr(7);
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::OUTPUT, 0);
    mixl2_task_ctx->add_task_addr(8);
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::OUTPUT, 1);
    mixl2_task_ctx->add_task_addr(9);
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::OUTPUT, 2);
    mixl2_task_ctx->add_task_addr(10);
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::WORKSPACE, 0);
    mixl2_task_ctx->add_task_addr(11);
    ge::ArgsFormatDescUtils::Append(args_desc, ge::AddrType::TILING_FFTS, 1);
    mixl2_task_ctx->add_task_addr(12);
    mixl2_task_ctx->set_args_format(ge::ArgsFormatDescUtils::ToString(args_desc));

    std::vector<domi::TaskDef> tasks;
    tasks.emplace_back(aicore_task);

    this->InitHolder(*node, "AlltoAllvGroupedMatMul");
    SetSpaceRegistry("AlltoAllvGroupedMatMul");
    GenerateOpExtTask(*node, this->holder, tasks);

    EXPECT_EQ(tasks.size(), 4U);
    EXPECT_EQ(tasks[0U].type(), RT_MODEL_TASK_NOTIFY_WAIT);
    EXPECT_EQ(tasks[1U].type(), RT_MODEL_TASK_KERNEL);
    EXPECT_EQ(tasks[2U].type(), RT_MODEL_TASK_NOTIFY_RECORD);
    EXPECT_EQ(tasks[3U].type(), RT_MODEL_TASK_FFTS_PLUS_TASK);

    const auto new_task_ctx = tasks[3].ffts_plus_task().ffts_plus_ctx(0).mix_aic_aiv_ctx();
    EXPECT_EQ(new_task_ctx.args_format(),
              "{ffts_addr}{hi.hcom0*}{i0*}{i1*}{i2*}{i3*}{i4*}{i5*}{o0*}{o1*}{o2*}{ws0*}{t_ffts.tail}");
    EXPECT_EQ(new_task_ctx.task_addr_size(), 13);
    EXPECT_EQ(new_task_ctx.task_addr(0), 1);
    EXPECT_EQ(new_task_ctx.task_addr(1), 0);
    EXPECT_EQ(new_task_ctx.task_addr(2), 2);
    EXPECT_EQ(new_task_ctx.task_addr(3), 3);
    EXPECT_EQ(new_task_ctx.task_addr(4), 4);
    EXPECT_EQ(new_task_ctx.task_addr(5), 5);
    EXPECT_EQ(new_task_ctx.task_addr(6), 6);
    EXPECT_EQ(new_task_ctx.task_addr(7), 7);
    EXPECT_EQ(new_task_ctx.task_addr(8), 8);
    EXPECT_EQ(new_task_ctx.task_addr(9), 9);
    EXPECT_EQ(new_task_ctx.task_addr(10), 10);
    EXPECT_EQ(new_task_ctx.task_addr(11), 11);
    EXPECT_EQ(new_task_ctx.task_addr(12), 12);
}
}  // namespace