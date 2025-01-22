#include <spconvlib/cumm/gemm/main/Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1/GemmKernel.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1 {
using TensorViewNVRTCKernel = spconvlib::cumm::common::TensorViewNVRTCKernel;
using RowMajor = spconvlib::cumm::gemm::layout::RowMajor;
using ColumnMajor = spconvlib::cumm::gemm::layout::ColumnMajor;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using GemmUtilsCPU = spconvlib::cumm::gemm::utils::GemmUtilsCPU;
using GemmKernelFlags = spconvlib::cumm::common::GemmKernelFlags;
using GemmUtils = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1::gemmutils::GemmUtils;
using InputIteratorA = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1::inpitera::MaskTileIterator;
using InputIteratorB = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1::inpiterb::MaskTileIterator;
using BlockMmaStorage = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1::gemm_smem_storage::BlockMmaStorage;
using OutputSmemStorage = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1::out_smem_storage::OutputSmemStorage;
using GemmParams = spconvlib::cumm::gemm::main::gpVolta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1::GemmParams;
using OutIter = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1::out_iter::OutIterator;
using ConstOutIter = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1::out_iter_const::OutIterator;
using OutputOp = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1::out_op::LinearCombination;
using Mma = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1::mma::Mma;
using Output = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1::output::Output;
__global__ void gemm_kernel(GemmParams params)   {
  
  #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
    // tv::printf2_once("?????222", params.grid_dims.x, params.grid_dims.y);
    constexpr bool kSplitKSerial = false;
    extern __shared__ uint8_t SharedStorage[];
    auto gemm_shared_mem =
        reinterpret_cast<BlockMmaStorage *>(SharedStorage);
    auto out_shared_mem =
        reinterpret_cast<OutputSmemStorage *>(SharedStorage);
    int tile_offset_m = blockIdx.x;
    int tile_offset_n = blockIdx.y;
    int tile_offset_k = blockIdx.z;
    if (tile_offset_m >= params.grid_dims.x || tile_offset_n >= params.grid_dims.y){
      return;
    }
    tv::array<int, 2> block_offset_A{tile_offset_m * 64,
                                    tile_offset_k * params.gemm_k_size_per_split};
    tv::array<int, 2> block_offset_B{tile_offset_k * params.gemm_k_size_per_split,
                                    tile_offset_n * 128};
    // Gemm::InputIteratorA::Params params_A(params.k);
    // Gemm::InputIteratorB::Params params_B(params.n);
    // refine gemm iteration for split-k
    auto problem_size_k = GemmUtils::get_gemm_k_bound(params.k, params.gemm_k_size_per_split, tile_offset_k);
    auto gemm_k_iterations = GemmUtils::get_gemm_iterations(problem_size_k, params.gemm_k_size_per_split, tile_offset_k);
    // int problem_size_k = min(params.k, (tile_offset_k + 1) * params.gemm_k_size_per_split);
    // int gemm_k_iterations =
    //     tv::div_up(problem_size_k - block_offset_A[1], 32);
    int thread_idx = threadIdx.x;
    InputIteratorA input_iter_A(
        params.itera_params_, params.ptr_A,
        tv::array<int, 2>{params.m, problem_size_k},
        thread_idx,
        block_offset_A);
    InputIteratorB input_iter_B(
        params.iterb_params_, params.ptr_B,
        tv::array<int, 2>{params.n, problem_size_k},
        thread_idx,
        tv::array<int, 2>{block_offset_B[1], block_offset_B[0]});
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;
    int warp_mn =
        warp_idx % (2 * 2);
    int warp_idx_k =
        warp_idx / (2 * 2);
    int warp_m = warp_mn % 2;
    int warp_n = warp_mn / 2;
    Mma mma(gemm_shared_mem, thread_idx, warp_idx_k, warp_m, warp_n, lane_idx);
    tv::array<tv::half_t, 64, 0> accumulators;
    accumulators.clear();
    if (!kSplitKSerial || gemm_k_iterations > 0){
      mma(gemm_k_iterations, accumulators, input_iter_A, input_iter_B, accumulators);
    }
    // tv::printf2_once("HERE 0", threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.z);
    // // C = alpha * A@B + beta * D, D can be C
    OutputOp output_op(params.alpha, params.beta, params.act_alpha, params.act_beta, params.act_type);
    tv::array<int, 2> block_offset_C{tile_offset_m * 64,
                                    tile_offset_n * 128};
    OutIter out_iter_C(params.out_params_, params.ptr_C, {params.m, params.n},
                            {block_offset_C[0], block_offset_C[1]},
                            thread_idx);
    ConstOutIter out_iter_D(params.out_params_scalebias_, params.ptr_D, {params.m, params.n},
                        {block_offset_C[0], block_offset_C[1]},
                        thread_idx);
    Output out(out_shared_mem, thread_idx, warp_idx_k, warp_m, warp_n, lane_idx);
    out.run(output_op, accumulators, out_iter_C, out_iter_D);
  #else
    tv::printf2_once("this arch isn't supported!");
    assert(0);
  #endif
}
} // namespace Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib