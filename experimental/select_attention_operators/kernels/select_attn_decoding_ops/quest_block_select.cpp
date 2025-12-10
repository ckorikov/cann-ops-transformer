#include "kernel_operator.h"

#define BYTES_UB_BLOCK 32
#define BYTES_DATA_BLOCK 32
#define NUM_PER_VECTOR 128
#define DIV_ROUNDUP(x,y) (((x)+(y)-1) / (y))
#define DIV_ROUNDUP_MUL(bytes,bytes_per_block) (DIV_ROUNDUP(bytes,bytes_per_block) * (bytes_per_block))
#define NUM_UB_BYTES(bytes) (DIV_ROUNDUP_MUL(bytes,BYTES_UB_BLOCK))
#define NUM_DATA_BLOCKS(bytes) (DIV_ROUNDUP(bytes,BYTES_DATA_BLOCK))
#define MIN(a,b) (((a)<(b))?(a):(b))

constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_V200 = 8;
constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_HALF_V220 = 4;
constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 = 2;

/**
 * @brief Kernel implementation for Quest KV block selection in AscendC
 *
 * @param [in] query Pointer to the query vector [B,H,D]
 * @param [in] maxblock Pointer to the quet metadata with the maximum vectors of every K block [B,N,BLOCK_SIZE,D]
 * @param [in] minblock Pointer to the quet metadata with the minimum vectors of every K block [B,N,BLOCK_SIZE,D]
 * @param [out] selected_indices Pointer to output indices vector [B,N,k] 
 * @param [in] B Batch size
 * @param [in] N Number of KV heads
 * @param [in] H Number of attention heads (query heads)
 * @param [in] BLOCK_SIZE Number of vectors in maxblock and in minblock
 * @param [in] D head dimension
 * @param [in] k number of top indices to return for every KV head
 */
extern "C" __global__ __aicore__ void quest_block_select(GM_ADDR query,
                                                         GM_ADDR maxblock,
                                                         GM_ADDR minblock,
                                                         GM_ADDR selected_indices,
                                                         int32_t B,
                                                         int32_t N,
                                                         int32_t H,
                                                         int32_t BLOCK_SIZE,
                                                         int32_t D,
                                                         int32_t k,
                                                         bool use_bfloat16)
{
    AscendC::SetAtomicNone();
    
    // Tile across AI cores - each core processes one batch*N combination
    int32_t num_blocks = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    int32_t num_batch_heads = B * N;
    int32_t num_batch_heads_per_block = DIV_ROUNDUP(num_batch_heads, num_blocks);
    int32_t G = H / N; // query_heads_per_kv_head

    // Init the GM pointers
    AscendC::GlobalTensor<half> query_gm;
    AscendC::GlobalTensor<half> maxblock_gm;
    AscendC::GlobalTensor<half> minblock_gm;
    AscendC::GlobalTensor<uint32_t> selected_indices_gm;
	AscendC::GlobalTensor<half> block_scores_gm;
    query_gm.SetGlobalBuffer((__gm__ half *)query);
    maxblock_gm.SetGlobalBuffer((__gm__ half *)maxblock);
    minblock_gm.SetGlobalBuffer((__gm__ half *)minblock);
    selected_indices_gm.SetGlobalBuffer((__gm__ uint32_t *)selected_indices);    

    // Allocate UB buffers
    using VecBuf_t = AscendC::TBuf<AscendC::QuePosition::VECCALC>;
    VecBuf_t query_buf, grouped_query_buf, maxblock_buf, minblock_buf, product_max_buf, product_min_buf;
    VecBuf_t channel_max_product_buf, block_scores_buf, selected_indices_buf, selected_values_buf;
    VecBuf_t tmp_concat_buf, concat_buf, index_local_buf, sort_tmp_buf;

    // Calculate buffer sizes (have to be runded up to multiple of BYTES_UB_BLOCK)
    uint32_t QUERY_BUF_SIZE = NUM_UB_BYTES(G * D * sizeof(half));
    uint32_t GROUPED_QUERY_SIZE = NUM_UB_BYTES(D * sizeof(half));
    uint32_t BLOCK_BUF_SIZE = NUM_UB_BYTES(BLOCK_SIZE * D * sizeof(half));
    uint32_t REDUCED_BUF_SIZE = NUM_UB_BYTES(BLOCK_SIZE * sizeof(half));
    uint32_t SELECTED_INDICES_BUF_SIZE = NUM_UB_BYTES(k * sizeof(uint32_t));
    uint32_t SELECTED_VALUES_BUF_SIZE = NUM_UB_BYTES(k * sizeof(half));
    uint32_t TMP_CONCAT_BUF_SIZE = NUM_UB_BYTES(BLOCK_SIZE * REGION_PROPOSAL_DATA_SIZE_V200 * sizeof(half));  // tmp_local size is "elemCount * REGION_PROPOSAL_DATA_SIZE_V200 * dataTypeSize;" accorsind to ascendc-src/tikcpp/tikcfw/lib/sort/sort_tiling_intf.h
    uint32_t CONCAT_BUF_SIZE = NUM_UB_BYTES((BLOCK_SIZE + BLOCK_SIZE * REGION_PROPOSAL_DATA_SIZE_V200) * sizeof(half));  // tmp_local size is "elemCount * REGION_PROPOSAL_DATA_SIZE_V200 * dataTypeSize;" accorsind to ascendc-src/tikcpp/tikcfw/lib/sort/sort_tiling_intf.h
    uint32_t INDEX_LOCAL_BUF_SIZE = NUM_UB_BYTES(BLOCK_SIZE * sizeof(uint32_t));
    uint32_t SORT_TMP_BUF_SIZE = NUM_UB_BYTES(BLOCK_SIZE * REGION_PROPOSAL_DATA_SIZE_HALF_V220 * sizeof(half));    

    // Initialize buffers
    AscendC::TPipe pipe;
    pipe.InitBuffer(query_buf, QUERY_BUF_SIZE);
    pipe.InitBuffer(grouped_query_buf, GROUPED_QUERY_SIZE);
    pipe.InitBuffer(maxblock_buf, BLOCK_BUF_SIZE);
    pipe.InitBuffer(minblock_buf, BLOCK_BUF_SIZE);
    pipe.InitBuffer(product_max_buf, BLOCK_BUF_SIZE);
    pipe.InitBuffer(product_min_buf, BLOCK_BUF_SIZE);
    pipe.InitBuffer(channel_max_product_buf, BLOCK_BUF_SIZE);
    pipe.InitBuffer(block_scores_buf, REDUCED_BUF_SIZE);
    pipe.InitBuffer(selected_indices_buf, SELECTED_INDICES_BUF_SIZE);
    pipe.InitBuffer(selected_values_buf, SELECTED_VALUES_BUF_SIZE);
    pipe.InitBuffer(tmp_concat_buf, TMP_CONCAT_BUF_SIZE);  
    pipe.InitBuffer(concat_buf, CONCAT_BUF_SIZE);  
    pipe.InitBuffer(index_local_buf, INDEX_LOCAL_BUF_SIZE);
    pipe.InitBuffer(sort_tmp_buf, SORT_TMP_BUF_SIZE);

    AscendC::LocalTensor<half> query_lt = query_buf.Get<half>();
    AscendC::LocalTensor<half> grouped_query_lt = grouped_query_buf.Get<half>();
    AscendC::LocalTensor<half> maxblock_lt = maxblock_buf.Get<half>();
    AscendC::LocalTensor<half> minblock_lt = minblock_buf.Get<half>();
    AscendC::LocalTensor<half> product_max_lt = product_max_buf.Get<half>();
    AscendC::LocalTensor<half> product_min_lt = product_min_buf.Get<half>();
    AscendC::LocalTensor<half> channel_max_product_lt = channel_max_product_buf.Get<half>();
    AscendC::LocalTensor<half> block_scores_lt = block_scores_buf.Get<half>();
    AscendC::LocalTensor<uint32_t> selected_indices_lt = selected_indices_buf.Get<uint32_t>();
    AscendC::LocalTensor<half> selected_values_lt = selected_values_buf.Get<half>();
    AscendC::LocalTensor<half> tmp_concat_lt = tmp_concat_buf.Get<half>();  
    AscendC::LocalTensor<half> concat_lt = concat_buf.Get<half>();  
    AscendC::LocalTensor<uint32_t> index_local_lt = index_local_buf.Get<uint32_t>();
    AscendC::LocalTensor<half> sort_tmp_lt = sort_tmp_buf.Get<half>();

    for (int32_t batch_head_idx = AscendC::GetBlockIdx(); batch_head_idx < num_batch_heads; batch_head_idx += num_blocks) {

        int32_t batch_idx = batch_head_idx / N; 
        int32_t head_idx = batch_head_idx % N;  // kv-head aka kv-group index within the batch index
        int32_t query_head_start_idx = head_idx * G; // index of the first query-head of the current kv-head

        // Calculate GM offsets specific to the current batch_head_idx
        int32_t query_offset = batch_idx * H * D + query_head_start_idx * D;  // Start of batch in query
        int32_t block_offset = batch_head_idx * BLOCK_SIZE * D;  // Start of batch*head in maxblock/minblock
        int32_t output_offset = batch_head_idx * k;  // Start of output for this batch*head
        int32_t output_blockscores_offset = batch_head_idx * D;

        // Step 1: Reduce queries across H dimension to get grouped_query [B,N,D]
        // Since H heads are grouped to N heads, we reduce every H/N heads
        // 1.1: Copy query GM -> UB
        uint16_t query_copy_block_len = NUM_DATA_BLOCKS(G * D * sizeof(half));
        auto query_copy_params = AscendC::DataCopyParams(1, query_copy_block_len, 0, 0);
        AscendC::DataCopy(query_lt, query_gm[query_offset], query_copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        
        // Step 1.2: Reduce-Mean query across H dimension - simple averaging of H/N queries channel-wise
        uint64_t mask = D; // D must be 128, otherwise everything breaks
        AscendC::Copy(grouped_query_lt, query_lt, mask, 1, { 1, 1, 8, 8 }); // initialize accumulator with 1st query head
        AscendC::PipeBarrier<PIPE_V>();
        // https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0106.html
        for (int32_t qhead = 1; qhead < G; qhead++) {
            // https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0036.html
            AscendC::Add<half>(grouped_query_lt, grouped_query_lt, query_lt[qhead * D], D);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // Step 1.3: Normalize by the group size
        // https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0055.html
        half scale = (half)((float)1.0 / (float)G);

        AscendC::Muls<half>(grouped_query_lt, grouped_query_lt, scale, D);
        AscendC::PipeBarrier<PIPE_V>();


        // Step 2: Original quest prediction of top-k important blocks
        
        // Step 2.1: Copy maxblock and minblock to GM -> UB
        uint16_t block_copy_block_len = NUM_DATA_BLOCKS(BLOCK_SIZE * D * sizeof(half));
        auto block_copy_params = AscendC::DataCopyParams(1, block_copy_block_len, 0, 0);
        AscendC::DataCopy(maxblock_lt, maxblock_gm[block_offset], block_copy_params);
        AscendC::DataCopy(minblock_lt, minblock_gm[block_offset], block_copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        
        // Step 2.2: Elementwise multiply one query vector with all BLOCK_SIZE vectors of maxblock and minblock
        
        // repeatParams structure:
        // dstBlkStride, src0BlkStride, src1BlkStride = 1, Continuous data reads and writes within a single iteration
        // dstRepStride, src0RepStride, src1RepStride = 8, Continuous Data Reads and Writes Between Adjacent
        // https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0013.html
        uint8_t repeatTimes = BLOCK_SIZE;
        AscendC::BinaryRepeatParams repeatParams = {1,1,1,8,0,8}; // src0RepStride=0 will guarantee that query vector of length D=128 is multiplied by every 128 elements of minblock_lt 
        AscendC::Mul(product_max_lt, grouped_query_lt, maxblock_lt, mask, repeatTimes, repeatParams); 
        AscendC::Mul(product_min_lt, grouped_query_lt, minblock_lt, mask, repeatTimes, repeatParams); 

        // Step 2.3: Elementwise max between product_min and product_max
        AscendC::Max(channel_max_product_lt, product_max_lt, product_min_lt, mask, repeatTimes, {1,1,1,8,8,8});

        // Step 2.4: Reduce sum the last dimension (D to 1) for each block
        // Assuming D = 128, we can use efficient "RepeatReduceSum" vector operation that reduces each 128 numbers to 1
        // https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0086.html
        // dstRepStride is 1, srcBlkStride is 1, srcRepStride is 8, and dstBlkStride is invalid. 
		// void RepeatReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask, const int32_t dstBlkStride, const int32_t srcBlkStride, const int32_t dstRepStride, const int32_t srcRepStride);
        AscendC::RepeatReduceSum<half, true>(block_scores_lt, channel_max_product_lt, repeatTimes, mask, 0, 1, 1, 8);

        // Step 2.5: Find top-k indices

        ///////////////////////////Find top-k using AscendC Concat --> Sort --> Extract API ///////////////////////////////////////
        uint32_t m_elementCount = BLOCK_SIZE;
        uint32_t m_concatRepeatTimes = m_elementCount / 32;
        uint32_t m_sortRepeatTimes = m_elementCount / 32;
        uint32_t m_extractRepeatTimes = m_elementCount / 32;

        // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_0839.html
        // void Concat(LocalTensor<T> &concatLocal, const LocalTensor<T> &srcLocal, const LocalTensor<T> &tmpLocal, const int32_t repeatTimes)
        AscendC::Concat(concat_lt, block_scores_lt, tmp_concat_lt, m_concatRepeatTimes);

        // create index range [0...BLOCK_SIZE-1]
        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {index_local_lt.SetValue(i,i);} // couldn't use CreateVecIndex(), as it doesn't suport uint32_t

        // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_0842.html
        // void Sort<T, isFullSort>(sortedLocal, concatLocal, indexLocal, sortTmpLocal, m_sortRepeatTimes);        
        AscendC::Sort<half, true>(product_max_lt, concat_lt, index_local_lt, sort_tmp_lt, m_sortRepeatTimes);
        
        // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_0841.html
        // void Extract(const LocalTensor<T> &dstValueLocal, const LocalTensor<uint32_t> &dstIndexLocal, const LocalTensor<T> &sortedLocal, const int32_t repeatTimes)
        AscendC::Extract(selected_values_lt, selected_indices_lt, product_max_lt, m_extractRepeatTimes);

        // Step 3: Copy out the results (UB -> GM)
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        uint16_t indices_copy_block_len = NUM_DATA_BLOCKS(k * sizeof(int32_t));
        auto indices_copy_params = AscendC::DataCopyParams(1, indices_copy_block_len, 0, 0);
        AscendC::DataCopy(selected_indices_gm[output_offset], selected_indices_lt, indices_copy_params);

    }
}

/**
 * Kernel launch function
 */
void launch_quest_block_select(
    uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *query,
    uint8_t *maxblock,
    uint8_t *minblock,
    uint8_t *selected_indices,
    int32_t B,
    int32_t N,
    int32_t H,
    int32_t BLOCK_SIZE,
    int32_t D,
    int32_t k,
    bool use_bfloat16)
{
    quest_block_select<<<blockDim, l2ctrl, stream>>>(
        query,
        maxblock,
        minblock,
        selected_indices,
        B, N, H, BLOCK_SIZE, D, k, use_bfloat16);
}
