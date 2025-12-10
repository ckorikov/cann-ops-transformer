#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "acl/acl.h"

#define MAX(a,b) (((a)>(b))?(a):(b))
#define DIV_ROUNDUP_MUL(x,y) ((((x)+(y)-1) / (y)) * (y))
#define BYTES_PER_IDX 4 // uint32_t
#define BYTES_PER_VAL 2 // half
#define BYTES_ASCEND_DATA_BLOCK 32

// Helper function to check if tensor is bfloat16
bool is_bfloat16(const at::Tensor& tensor) {
    return tensor.scalar_type() == at::kBFloat16;
}

/*****************************************************************************/
/*** Paged version (multiple metadata blocks per request, traveresed using ***/
/*** metadata_block_tables)                                                ***/
/*****************************************************************************/
void launch_quest_block_select_paged(
    uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *query,
    uint8_t *maxblocks,
    uint8_t *minblocks,
    uint8_t *metadata_block_tables,
    uint8_t *seq_lens,
    uint8_t *selected_indices,
    int32_t B,
    int32_t N,
    int32_t H,
    int32_t BLOCK_SIZE,
    int32_t D,
    int32_t MMBPR,
    int32_t num_meta_blocks,
    int32_t k,
    bool use_bfloat16
);

/**
 * @brief Interface the `quest_block_select_paged` kernel which predicts the
 *        sparsity mask during decoding in the form of top-k important kv-block 
 *        indices for every KV-head in every request. The returned KV block ids 
 *        are not the indices in the KV-cache, but rather from their enumeration 
 *        from 0 to number of blocks in the sequence length being decoded.
 *
 * @param [in] query Tensor(fp16) the query vector [B,H,D]
 * @param [in] maxblocks Tensor(fp16) quest metadata with the maximum vectors 
 *                       of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 * @param [in] minblocks Tensor(fp16) quest metadata with the minimum vectors 
 *                       of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 * @param [in] metadata_block_tables Tensor(int32) the metadata block tables [B,MMBPR]
 * @param [in] seq_lens Tensor(int32) sequence length of each request in the batch [B]
 * @param [in] k natural number of highest indices to return for every KV head
 * 
 * @returns selected_indices - Tensor(int32) where the kernel will output the selected 
 *          indices vector. [B,N,k] 
 */
at::Tensor quest_block_select_paged(at::Tensor query,
                                    at::Tensor maxblocks,
                                    at::Tensor minblocks,
                                    at::Tensor metadata_block_tables,
                                    at::Tensor seq_lens,
                                    int k
)
{
// round the k to a number that will require a multiple of 32 bytes
    int k_round = DIV_ROUNDUP_MUL(k * BYTES_PER_IDX, BYTES_ASCEND_DATA_BLOCK) / BYTES_PER_IDX;

    // infer tensor shapes
    int32_t B = query.sizes()[0]; 
    int32_t N = maxblocks.sizes()[2];
    int32_t H = query.sizes()[1];
    int32_t BLOCK_SIZE = maxblocks.sizes()[1];
    int32_t D = query.sizes()[2];
    int32_t MMBPR = metadata_block_tables.sizes()[1];
    int32_t num_meta_blocks = maxblocks.sizes()[0];
    
    // validate input shapes
    TORCH_CHECK(D == 128, "D must be equal to 128 for high performance operations, got ", D);
    TORCH_CHECK(BLOCK_SIZE == 128, "BLOCK_SIZE must be equal to 128 for high performance operations, got ", BLOCK_SIZE);
    TORCH_CHECK(H % N == 0, "H must be divisible by N (GQA/MHA requirement). H=", H, " N=", N);
    TORCH_CHECK(B == seq_lens.sizes()[0], "Batch size mismatch: query batch=", B, " seq_lens batch=", seq_lens.size(0));
    TORCH_CHECK(B == metadata_block_tables.size(0), "Batch size mismatch: query batch=", B, " metadata_block_tables batch=", metadata_block_tables.size(0));
    TORCH_CHECK(N == minblocks.size(2), "N (num KV heads) mismatch: expected ", N, " from query, got ", minblocks.size(2), " from minblocks");
    TORCH_CHECK(BLOCK_SIZE == minblocks.size(1), "BLOCK_SIZE mismatch: expected ", BLOCK_SIZE, " , got ", minblocks.size(1), " from minblocks");
    TORCH_CHECK(D == maxblocks.size(3), "Head dimension D mismatch: expected ", D, " from query, got ", maxblocks.size(3), " from maxblocks");
    TORCH_CHECK(D == minblocks.size(3), "Head dimension D mismatch: expected ", D, " from query, got ", minblocks.size(3), " from minblocks");
    TORCH_CHECK(num_meta_blocks == minblocks.size(0), "num_meta_blocks mismatch: inferred ", num_meta_blocks, " from maxblocks, got ", minblocks.size(0), " from minblocks");
    TORCH_CHECK(k > 0, "k must be positive, got ", k);
    TORCH_CHECK(MMBPR < 7, "maximum metablocks per request (MMBPR) cannot exceed 6 for this kernel. Your MMBPR=", MMBPR, " as inferred from dim=1 of metadata_block_tables argument.");
    TORCH_CHECK(H / N <= BLOCK_SIZE, "H/N head group size cannot exceed BLOCK_SIZE=",BLOCK_SIZE, " given H/N=", H/N);

    // validate data types
    bool use_bfloat16 = is_bfloat16(query);
    TORCH_CHECK(use_bfloat16 == is_bfloat16(maxblocks), "query, maxblocks, minblocks input tensors must have the same data type");
    TORCH_CHECK(use_bfloat16 == is_bfloat16(minblocks), "query, maxblocks, minblocks input tensors must have the same data type");

    // allocate output tensor
    auto output_tensor_options = at::TensorOptions(query.options()).dtype(at::kInt);
    at::Tensor selected_indices = torch::empty({B, N, k_round}, output_tensor_options); 
    uint8_t *selected_indices_ptr = reinterpret_cast<uint8_t *>(selected_indices.storage().data_ptr().get());
    
    // allocate input tensors
    uint8_t *query_ptr = reinterpret_cast<uint8_t *>(query.storage().data_ptr().get());
    uint8_t *maxblocks_ptr = reinterpret_cast<uint8_t *>(maxblocks.storage().data_ptr().get());
    uint8_t *minblocks_ptr = reinterpret_cast<uint8_t *>(minblocks.storage().data_ptr().get());
    uint8_t *metadata_block_tables_ptr = reinterpret_cast<uint8_t *>(metadata_block_tables.storage().data_ptr().get());
    uint8_t *seq_lens_ptr = reinterpret_cast<uint8_t *>(seq_lens.storage().data_ptr().get());

    // set up launch parameters
    uint32_t blockDims = (B * N > NUM_CORES) ? NUM_CORES : B * N;
    int deviceId;
    aclrtGetDevice(&deviceId);
    auto npuStream = c10_npu::getCurrentNPUStream(deviceId);
    auto aclStream = npuStream.stream();

    // launch the kernel
    launch_quest_block_select_paged(
        blockDims, nullptr, aclStream,
        query_ptr,
        maxblocks_ptr,
        minblocks_ptr,
        metadata_block_tables_ptr,
        seq_lens_ptr,
        selected_indices_ptr,
        B,
        N,
        H,
        BLOCK_SIZE,
        D,
        MMBPR,
        num_meta_blocks,
        k_round,
        use_bfloat16
    );

    if (k != k_round) {
        // Trim the tensors to the original k size before returning
        selected_indices = selected_indices.slice(/*dim=*/2, /*start=*/0, /*end=*/k);
    }

    return selected_indices;
}


/*****************************************************************************/
/*** Non-Paged version (single metadata block per request, densely packed) ***/
/*****************************************************************************/
extern void launch_quest_block_select(
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
    bool use_bfloat16
);


/**
 * This is the interface function which is invoked from the python level
 * It handles
 *  1. Passing the pointers of the input tensors to the kernel
 *  2. Allocatin global memory for the output tensor
 *  3. Invocation of the kernel (which fills up the output tensor with a correct data)
 *  4. Returning the output tensor
 * @brief Interface the `quest_block_select` kernel (single block version), 
 *        which predicts the sparsity mask during decoding in the form of 
 *        top-k important kv-block indices for every KV-head in every request.
 *        The returned KV block ids are not the indices in the KV-cache, 
 *        but rather from their enumeration from 0 to number of blocks in the 
 *        sequence length being decoded.
 * @param [in] query Tensor(fp16 or bf16) the query vector [B,H,D]
 * @param [in] maxblock Tensor(fp16 or bf16) quest metadata with the maximum vectors of 
 *                      every K block [B,N,BLOCK_SIZE,D]
 * @param [in] minblock Tensor(fp16 or bf16) quest metadata with the minimum vectors of 
 *                      every K block [B,N,BLOCK_SIZE,D]
 * @param [in] k natural number of highest indices to return for every KV head
 * @returns selected_indices Tensor(int32) where the kernel will output the 
 *          selected indices vector. [B,N,k] 
 */
at::Tensor quest_block_select(at::Tensor query,
                              at::Tensor maxblock,
                              at::Tensor minblock,
                              int k
)
{
    // round the k to a number that will require a multiple of 32 bytes
    int k_round = DIV_ROUNDUP_MUL(k * BYTES_PER_IDX, BYTES_ASCEND_DATA_BLOCK) / BYTES_PER_IDX;

    // infer tensor shapes
    int32_t B = query.sizes()[0]; 
    int32_t N = maxblock.sizes()[1];
    int32_t H = query.sizes()[1];
    int32_t BLOCK_SIZE = maxblock.sizes()[2];
    int32_t D = query.sizes()[2];

    // validate input shapes
    TORCH_CHECK(D == 128, "D must be equal to 128 for high performance operations, got ", D);
    TORCH_CHECK(BLOCK_SIZE == 128, "BLOCK_SIZE must be equal to 128 for high performance operations, got ", BLOCK_SIZE);
    TORCH_CHECK(H % N == 0, "H must be divisible by N (GQA/MHA requirement). H=", H, " N=", N);
    TORCH_CHECK(B == maxblock.size(0), "Batch size mismatch: query batch=", B, " maxblock batch=", maxblock.size(0));
    TORCH_CHECK(B == minblock.size(0), "Batch size mismatch: query batch=", B, " minblock batch=", minblock.size(0));
    TORCH_CHECK(N == minblock.size(1), "N (num KV heads) mismatch: expected ", N, " from query, got ", minblock.size(1), " from minblock");
    TORCH_CHECK(BLOCK_SIZE == minblock.size(2), "BLOCK_SIZE mismatch: expected ", BLOCK_SIZE, ", got ", minblock.size(2), " from minblock");
    TORCH_CHECK(D == maxblock.size(3), "Head dimension D mismatch: expected ", D, " from query, got ", maxblock.size(3), " from maxblock");
    TORCH_CHECK(D == minblock.size(3), "Head dimension D mismatch: expected ", D, " from query, got ", minblock.size(3), " from minblock");
    TORCH_CHECK(k > 0, "k must be positive, got ", k);
    // validate data types
    bool use_bfloat16 = is_bfloat16(query);
    TORCH_CHECK(use_bfloat16 == is_bfloat16(maxblock), "All input tensors must have the same data type");
    TORCH_CHECK(use_bfloat16 == is_bfloat16(minblock), "All input tensors must have the same data type");
    TORCH_CHECK(!use_bfloat16, "bfloat16 datatype is not yet supported for quest_block_select");
    
    // allocate output tensor
    auto output_tensor_options = at::TensorOptions(query.options()).dtype(at::kInt);
    at::Tensor selected_indices = torch::empty({B, N, k_round}, output_tensor_options); 
    uint8_t *selected_indices_ptr = reinterpret_cast<uint8_t *>(selected_indices.storage().data_ptr().get());
    
    // allocate input tensors
    uint8_t *query_ptr = reinterpret_cast<uint8_t *>(query.storage().data_ptr().get());
    uint8_t *maxblock_ptr = reinterpret_cast<uint8_t *>(maxblock.storage().data_ptr().get());
    uint8_t *minblock_ptr = reinterpret_cast<uint8_t *>(minblock.storage().data_ptr().get());

    // set up launch parameters
    uint32_t blockDims = (B * N > NUM_CORES) ? NUM_CORES : B * N;
    int deviceId;
    aclrtGetDevice(&deviceId);
    auto npuStream = c10_npu::getCurrentNPUStream(deviceId);
    auto aclStream = npuStream.stream();

    // launch the kernel
    launch_quest_block_select(
        blockDims, nullptr, aclStream,
        query_ptr,
        maxblock_ptr,
        minblock_ptr,
        selected_indices_ptr,
        B,
        N,
        H,
        BLOCK_SIZE,
        D,
        k_round,
        use_bfloat16
    );

    if (k != k_round) {
        // Trim the tensors to the original k size before returning
        selected_indices = selected_indices.slice(/*dim=*/2, /*start=*/0, /*end=*/k);
    }

    return selected_indices;
}

/**
 * Create the binding between this CPP function and python. Expose the function towards python.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quest_block_select", &quest_block_select, 
        R"DOC(
        Interface to the `quest_block_select` kernel (single block version), 
        which predicts the sparsity mask during decoding in the form of 
        top-k important kv-block indices for every KV-head in every request.
        The returned KV block ids are not the indices in the KV-cache, 
        but rather from their enumeration from 0 to number of blocks in the sequence length 
        being decoded.

        Args:
            query (torch.Tensor): Query vector of shape [B, H, D] (fp16 or bf16)
            maxblock (torch.Tensor): Quest metadata with maximum vectors of 
                                   every K block of shape [B, N, BLOCK_SIZE, D] (fp16 or bf16)
            minblock (torch.Tensor): Quest metadata with minimum vectors of 
                                   every K block of shape [B, N, BLOCK_SIZE, D] (fp16 or bf16)
            k (int): Number of highest indices to return for every KV head

        Returns:
            torch.Tensor: Selected indices vector of shape [B, N, k] (int32)
        )DOC",
        py::arg("query"),
        py::arg("maxblock"),
        py::arg("minblock"),
        py::arg("k")
    );

    m.def("quest_block_select_paged", &quest_block_select_paged,
        R"DOC(
        Interface to the `quest_block_select_paged` kernel which predicts the
        sparsity mask during decoding in the form of top-k important kv-block 
        indices for every KV-head in every request. The returned KV block ids 
        are not the indices in the KV-cache, but rather from their enumeration 
        from 0 to number of blocks in the sequence length being decoded.

        Args:
            query (torch.Tensor): Query vector of shape [B, H, D] (fp16 or bf16)
            maxblocks (torch.Tensor): Quest metadata with maximum vectors of 
                                    every key-cache block of shape 
                                    [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
            minblocks (torch.Tensor): Quest metadata with minimum vectors of 
                                    every key-cache block of shape 
                                    [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
            metadata_block_tables (torch.Tensor): Metadata block tables of 
                                                shape [B, MMBPR] (int32)
            seq_lens (torch.Tensor): Sequence length of each request in the batch
                                   of shape [B] (int32)
            k (int): Number of highest indices to return for every KV head

        Returns:
            torch.Tensor: Selected indices vector of shape [B, N, k] (int32)

        Limitations: due to kernel's internal buffer design on 910B:
            D = 128
            BLOCK_SIZE = 128
            H / N <= BLOCK_SIZE
            MMBPR < 7 (below 5 is the most stable)
        )DOC",
        py::arg("query"),
        py::arg("maxblocks"),
        py::arg("minblocks"),
        py::arg("metadata_block_tables"),
        py::arg("seq_lens"),
        py::arg("k")
    );
}
