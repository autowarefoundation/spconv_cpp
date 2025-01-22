#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasic.h>
#include <spconvlib/cumm/gemm/utils/GemmUtilsCPU.h>
#include <spconvlib/cumm/gemm/main/gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1/gemmutils/GemmUtils.h>
#include <spconvlib/cumm/gemm/main/gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1/itera_p/MaskTileIteratorParams.h>
#include <spconvlib/cumm/gemm/main/gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1/iterb_p/MaskTileIteratorParams.h>
#include <spconvlib/cumm/gemm/main/gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1/out_params_ns/OutIteratorParams.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1 {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmUtilsCPU = spconvlib::cumm::gemm::utils::GemmUtilsCPU;
using GemmUtils = spconvlib::cumm::gemm::main::gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1::gemmutils::GemmUtils;
using IterAParams = spconvlib::cumm::gemm::main::gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1::itera_p::MaskTileIteratorParams;
using IterBParams = spconvlib::cumm::gemm::main::gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1::iterb_p::MaskTileIteratorParams;
using OutIterParams = spconvlib::cumm::gemm::main::gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1::out_params_ns::OutIteratorParams;
struct GemmParams {
  int gemm_k_size_per_split;
  int m;
  int n;
  int k;
  const tv::half_t* ptr_A;
  const tv::half_t* ptr_B;
  tv::half_t* ptr_C;
  const tv::half_t* ptr_D;
  int64_t stride_A;
  int64_t stride_B;
  int64_t stride_C;
  int64_t stride_D;
  tv::half_t alpha;
  tv::half_t beta;
  tv::half_t act_alpha;
  tv::half_t act_beta;
  tv::gemm::Activation act_type;
  dim3 grid_dims;
  IterAParams itera_params_;
  IterBParams iterb_params_;
  OutIterParams out_params_;
  OutIterParams out_params_scalebias_;
  __host__ __device__ inline  GemmParams(int m, int n, int k, const tv::half_t* A, const tv::half_t* B, tv::half_t* C, const tv::half_t* D, int64_t stride_A, int64_t stride_B, int64_t stride_C, int64_t stride_D, const int* IndiceA, const int* IndiceC, const int* IndiceD, tv::half_t alpha = tv::half_t(1), tv::half_t beta = tv::half_t(0), tv::half_t act_alpha = tv::half_t(0), tv::half_t act_beta = tv::half_t(0), tv::gemm::Activation act_type = tv::gemm::Activation::kNone, int split_k_slices = 1) : m(m), n(n), k(k), ptr_A(A), ptr_B(B), ptr_C(C), ptr_D(D), stride_A(stride_A), stride_B(stride_B), stride_C(stride_C), stride_D(stride_D), alpha(alpha), beta(beta), act_alpha(act_alpha), act_beta(act_beta), act_type(act_type)  {
    
    
    auto grid_dims_arr = GemmUtilsCPU::get_logical_tile_count(m, n, k, 64, 64, split_k_slices);
    // tv::printf2_once(m, n, k, 64, 64, split_k_slices, grid_dims_arr[0], grid_dims_arr[1], grid_dims_arr[2]);
    grid_dims.x = grid_dims_arr[0];
    grid_dims.y = grid_dims_arr[1];
    grid_dims.z = grid_dims_arr[2];
    // int total_gemm_k_iterations = tv::div_up(k, 64); // 160, 16 = 10
    // int gemm_k_iterations_per_split =
    //     tv::div_up(total_gemm_k_iterations, int(grid_dims.z)); // 10, 4 = 3
    // gemm_k_size_per_split = gemm_k_iterations_per_split * 64; // 3 * 16 = 48, 0-48, 48-96, 96-144, 144-192
    gemm_k_size_per_split = GemmUtils::get_gemm_k_size_per_split(k, split_k_slices);
    // tv::ssprint("gemm_k_size_per_split", m, n, k, gemm_k_size_per_split, grid_dims.x, grid_dims.y, grid_dims.z);
    itera_params_ = IterAParams(stride_A, IndiceA);
    iterb_params_ = IterBParams(stride_B);
    out_params_ = OutIterParams(stride_C, IndiceC);
    out_params_scalebias_ = OutIterParams(stride_D, IndiceD);
  }
};
} // namespace gpTuring_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib