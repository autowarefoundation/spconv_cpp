#include <spconvlib/cumm/conv/main/Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8/ConvKernel.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8 {
using TensorViewNVRTCKernel = spconvlib::cumm::common::TensorViewNVRTCKernel;
using RowMajor = spconvlib::cumm::gemm::layout::RowMajor;
using ColumnMajor = spconvlib::cumm::gemm::layout::ColumnMajor;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmKernelFlags = spconvlib::cumm::common::GemmKernelFlags;
using InputIteratorA = spconvlib::cumm::conv::main::Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::inpitera::ForwardDgradSparseIOIterator;
using InputIteratorB = spconvlib::cumm::conv::main::Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::inpiterb::WeightIteratorDP4A;
using LayoutA = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::la::TensorGeneric;
using LayoutB = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::lb::TensorGeneric;
using LayoutC = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::lc::TensorGeneric;
using BlockMmaStorage = spconvlib::cumm::conv::main::Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::gemm_smem_storage::BlockMmaStorage;
using OutputSmemStorage = spconvlib::cumm::conv::main::Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::out_smem_storage::OutputSmemStorage;
using ConvParams = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::ConvParams;
using ConvProblem = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::cp::ConvProblem;
using OutIter = spconvlib::cumm::conv::main::Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::out_iter::OutIterator;
using ConstOutIter = spconvlib::cumm::conv::main::Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::out_iter_const::OutIterator;
using ConstScaleOutIter = spconvlib::cumm::conv::main::Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::scale_out_iter_const::OutIterator;
using OutputOp = spconvlib::cumm::conv::main::Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::out_op::Int8Inference;
using Mma = spconvlib::cumm::conv::main::Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::mma::Mma;
using MaskIGemmIteratorDynamic = spconvlib::cumm::conv::main::Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::mma_miterd::MaskIGemmIteratorMaskLoaderDynamic;
using Output = spconvlib::cumm::conv::main::Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::output::Output;
__global__ void conv_kernel(ConvParams params)   {
  
  #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
    constexpr bool kSplitKSerial = false;
    extern __shared__ uint8_t SharedStorage[];
    auto gemm_shared_mem =
        reinterpret_cast<BlockMmaStorage *>(SharedStorage);
    auto out_shared_mem =
        reinterpret_cast<OutputSmemStorage *>(SharedStorage);
    int tile_offset_m = blockIdx.x;
    int tile_offset_n = blockIdx.y;
    int tile_offset_k = blockIdx.z;
    if (tile_offset_m >= params.grid_dims.x ||
        tile_offset_n >= params.grid_dims.y) {
        return;
    }
    tv::array<int, 2> block_offset_A{tile_offset_m * 64, tile_offset_k * 32};
    tv::array<int, 2> block_offset_B{tile_offset_n * 64, tile_offset_k * 32};
    int thread_idx = threadIdx.x;
    InputIteratorA input_iter_A(
        params.itera_params_, params.problem, params.ptr_A,
        thread_idx,
        block_offset_A);
    InputIteratorB input_iter_B(
        params.iterb_params_, params.problem, params.ptr_B,
        thread_idx,
        block_offset_B);
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;
    int warp_mn =
        warp_idx % (2 * 2);
    int warp_idx_k =
        warp_idx / (2 * 2);
    int warp_m = warp_mn % 2;
    int warp_n = warp_mn / 2;
    int mask_int_count = tv::div_up(params.problem.kernel_volume, 32);
    MaskIGemmIteratorDynamic MaskLoader(params.mask_ptr, 
                                        params.mask_out_ptr, 
                                        mask_int_count,
                                        tile_offset_m, params.gemm_k_iterations,
                                        params.problem.kernel_volume, params.mask_filter,
                                        false, 
                                        lane_idx, params.m);
    Mma mma(gemm_shared_mem, thread_idx, warp_idx_k, warp_m, warp_n, lane_idx);
    tv::array<int32_t, 32, 0> accumulators;
    accumulators.clear();
    if (!kSplitKSerial || params.gemm_k_iterations > 0){
      if (!MaskLoader.empty()){
          mma(params.gemm_k_iterations, accumulators, input_iter_A, input_iter_B, accumulators, MaskLoader, params.problem.kernel_volume);
      }
    }
    // // C = alpha * A@B + beta * D, D can be C
    OutputOp output_op(params.alpha, params.beta, params.act_alpha, params.act_beta, params.act_type);
    tv::array<int, 2> block_offset_C{tile_offset_m * 64,
                                    tile_offset_n * 64};
    tv::array<int, 2> block_extent_C{params.m, params.n};
    OutIter out_iter_C(params.out_params_, params.ptr_C, params.ptr_D, block_extent_C,
                            block_offset_C,
                            thread_idx);
    ConstScaleOutIter out_iter_bias(params.out_params_scalebias_, params.bias_pointer, block_extent_C,
                            block_offset_C,
                            thread_idx);
    ConstScaleOutIter out_iter_scale(params.out_params_scalebias_, params.scale_pointer, block_extent_C,
                            block_offset_C,
                            thread_idx);
    Output out(out_shared_mem, thread_idx, warp_idx_k, warp_m, warp_n, lane_idx);
    out.run(output_op, accumulators, out_iter_C, out_iter_bias, out_iter_scale);
  #else
    tv::printf2_once("this arch isn't supported!");
    assert(0);
  #endif
}
} // namespace Turing_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib