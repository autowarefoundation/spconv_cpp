#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasic.h>
#include <spconvlib/cumm/conv/kernel/ConvUtils.h>
#include <spconvlib/cumm/gemm/utils/GemmUtilsCPU.h>
#include <spconvlib/cumm/conv/main/cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK/cp/ConvProblem.h>
#include <spconvlib/cumm/conv/main/cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK/itera_p/SparseParams.h>
#include <spconvlib/cumm/conv/main/cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK/iterb_p/WeightOptParams.h>
#include <spconvlib/cumm/conv/main/cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK/out_params_ns/OutIteratorParams.h>
#include <spconvlib/cumm/conv/main/cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK/out_params_scalebias_ns/OutIteratorParams.h>
#include <spconvlib/cumm/conv/main/cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK/la/TensorGeneric.h>
#include <spconvlib/cumm/conv/main/cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK/lb/TensorGeneric.h>
#include <spconvlib/cumm/conv/main/cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK/lc/TensorGeneric.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using ConvUtils = spconvlib::cumm::conv::kernel::ConvUtils;
using GemmUtilsCPU = spconvlib::cumm::gemm::utils::GemmUtilsCPU;
using ConvProblem = spconvlib::cumm::conv::main::cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK::cp::ConvProblem;
using IterAParams = spconvlib::cumm::conv::main::cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK::itera_p::SparseParams;
using IterBParams = spconvlib::cumm::conv::main::cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK::iterb_p::WeightOptParams;
using OutIterParams = spconvlib::cumm::conv::main::cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK::out_params_ns::OutIteratorParams;
using OutIterParamsScaleBias = spconvlib::cumm::conv::main::cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK::out_params_scalebias_ns::OutIteratorParams;
using LayoutA = spconvlib::cumm::conv::main::cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK::la::TensorGeneric;
using LayoutB = spconvlib::cumm::conv::main::cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK::lb::TensorGeneric;
using LayoutC = spconvlib::cumm::conv::main::cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK::lc::TensorGeneric;
struct ConvParams {
  ConvProblem problem;
  int m;
  int n;
  int k;
  int gemm_k_iterations;
  const tv::half_t* ptr_A;
  const tv::half_t* ptr_B;
  tv::half_t* ptr_C;
  const tv::half_t* ptr_D;
  const uint32_t* mask_ptr;
  uint32_t* mask_out_ptr;
  uint32_t mask_filter;
  bool reverse_mask;
  tv::half_t alpha;
  tv::half_t beta;
  tv::half_t act_alpha;
  tv::half_t act_beta;
  tv::gemm::Activation act_type;
  dim3 grid_dims;
  IterAParams itera_params_;
  IterBParams iterb_params_;
  OutIterParams out_params_;
  OutIterParams out_params_source_;
  /**
   * @param problem 
   * @param A 
   * @param B 
   * @param C 
   * @param D 
   * @param mask_ptr 
   * @param mask_argsort_ptr 
   * @param indice_ptr 
   * @param mask_out_ptr 
   * @param mask_filter 
   * @param reverse_mask 
   * @param alpha 
   * @param beta 
   * @param act_alpha 
   * @param act_beta 
   * @param act_type 
   * @param split_k_slices 
   * @param d_is_bias 
   */
  __host__ __device__  ConvParams(ConvProblem problem, const tv::half_t* A, const tv::half_t* B, tv::half_t* C, const tv::half_t* D, const uint32_t* mask_ptr, const int* mask_argsort_ptr, const int* indice_ptr, uint32_t* mask_out_ptr, uint32_t mask_filter, bool reverse_mask, tv::half_t alpha = tv::half_t(1), tv::half_t beta = tv::half_t(0), tv::half_t act_alpha = tv::half_t(0), tv::half_t act_beta = tv::half_t(0), tv::gemm::Activation act_type = tv::gemm::Activation::kNone, int split_k_slices = 1, bool d_is_bias = false);
};
} // namespace cpVolta_f16f16f16f16f16tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib