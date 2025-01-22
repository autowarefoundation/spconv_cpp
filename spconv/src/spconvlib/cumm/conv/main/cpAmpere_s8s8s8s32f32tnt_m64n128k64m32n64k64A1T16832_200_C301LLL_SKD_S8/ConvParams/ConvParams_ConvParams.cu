#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8/ConvParams.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace cpAmpere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8 {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using ConvUtils = spconvlib::cumm::conv::kernel::ConvUtils;
using GemmUtilsCPU = spconvlib::cumm::gemm::utils::GemmUtilsCPU;
using ConvProblem = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8::cp::ConvProblem;
using IterAParams = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8::itera_p::SparseParams;
using IterBParams = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8::iterb_p::WeightOptParams;
using OutIterParams = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8::out_params_ns::OutIteratorParams;
using OutIterParamsScaleBias = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8::out_params_scalebias_ns::OutIteratorParams;
using LayoutA = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8::la::TensorGeneric;
using LayoutB = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8::lb::TensorGeneric;
using LayoutC = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8::lc::TensorGeneric;
__host__ __device__  ConvParams::ConvParams(ConvProblem problem, const int8_t* A, const int8_t* B, int8_t* C, const int8_t* D, const float* bias_pointer, const float* scale_pointer, const uint32_t* mask_ptr, const int* mask_argsort_ptr, const int* indice_ptr, uint32_t* mask_out_ptr, uint32_t mask_filter, bool reverse_mask, float alpha, float beta, float act_alpha, float act_beta, tv::gemm::Activation act_type, int split_k_slices, bool d_is_bias) : problem(problem), itera_params_(problem, indice_ptr, mask_argsort_ptr), mask_out_ptr(mask_out_ptr), iterb_params_(problem, LayoutB::from_shape(problem.get_weight_shape())), ptr_A(A), ptr_B(B), ptr_C(C), ptr_D(D), mask_ptr(mask_ptr), mask_filter(mask_filter), reverse_mask(reverse_mask), alpha(alpha), beta(beta), act_alpha(act_alpha), act_beta(act_beta), act_type(act_type), bias_pointer(bias_pointer), scale_pointer(scale_pointer)  {
  
  auto mnk = problem.implicit_gemm_mnk(tv::gemm::ConvOpType::kForward);
  m = mnk[0];
  n = mnk[1];
  k = mnk[2];
  gemm_k_iterations = problem.implicit_gemm_k_iterations(tv::gemm::ConvOpType::kForward, 64);
  #if !defined(__CUDACC_RTC__) && !defined(__NVCC__)
  TV_ASSERT_INVALID_ARG(gemm_k_iterations % problem.kernel_volume == 0, "error");
  #endif
  auto grid_dims_arr = ConvUtils::get_spconv_logical_tile_count(m, n, k, 
      64, 128, split_k_slices, problem.kernel_volume, 0);
  grid_dims.x = grid_dims_arr[0];
  grid_dims.y = grid_dims_arr[1];
  grid_dims.z = grid_dims_arr[2];
  gemm_k_iterations /= problem.kernel_volume;
  itera_params_.set_inc_reset_for_inc_k_first(gemm_k_iterations);
  iterb_params_.set_inc_reset_for_inc_k_first(gemm_k_iterations);
  out_params_ = OutIterParams(n, mask_argsort_ptr);
  out_params_scalebias_ = OutIterParamsScaleBias();
}
} // namespace cpAmpere_s8s8s8s32f32tnt_m64n128k64m32n64k64A1T16832_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib