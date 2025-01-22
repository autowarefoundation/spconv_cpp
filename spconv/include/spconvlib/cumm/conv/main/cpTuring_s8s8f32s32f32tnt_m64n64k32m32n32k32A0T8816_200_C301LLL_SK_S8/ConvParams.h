#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasic.h>
#include <spconvlib/cumm/conv/kernel/ConvUtils.h>
#include <spconvlib/cumm/gemm/utils/GemmUtilsCPU.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/cp/ConvProblem.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/itera_p/SparseParams.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/iterb_p/WeightOptParams.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/out_params_ns/OutIteratorParams.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/out_params_scalebias_ns/OutIteratorParams.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/la/TensorGeneric.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/lb/TensorGeneric.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/lc/TensorGeneric.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8 {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using ConvUtils = spconvlib::cumm::conv::kernel::ConvUtils;
using GemmUtilsCPU = spconvlib::cumm::gemm::utils::GemmUtilsCPU;
using ConvProblem = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::cp::ConvProblem;
using IterAParams = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::itera_p::SparseParams;
using IterBParams = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::iterb_p::WeightOptParams;
using OutIterParams = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::out_params_ns::OutIteratorParams;
using OutIterParamsScaleBias = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::out_params_scalebias_ns::OutIteratorParams;
using LayoutA = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::la::TensorGeneric;
using LayoutB = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::lb::TensorGeneric;
using LayoutC = spconvlib::cumm::conv::main::cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::lc::TensorGeneric;
struct ConvParams {
  ConvProblem problem;
  int m;
  int n;
  int k;
  int gemm_k_iterations;
  const int8_t* ptr_A;
  const int8_t* ptr_B;
  float* ptr_C;
  const float* ptr_D;
  const float* bias_pointer;
  const float* scale_pointer;
  const uint32_t* mask_ptr;
  uint32_t* mask_out_ptr;
  uint32_t mask_filter;
  bool reverse_mask;
  float alpha;
  float beta;
  float act_alpha;
  float act_beta;
  tv::gemm::Activation act_type;
  dim3 grid_dims;
  IterAParams itera_params_;
  IterBParams iterb_params_;
  OutIterParams out_params_;
  OutIterParamsScaleBias out_params_scalebias_;
  /**
   * @param problem 
   * @param A 
   * @param B 
   * @param C 
   * @param D 
   * @param bias_pointer 
   * @param scale_pointer 
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
  __host__ __device__  ConvParams(ConvProblem problem, const int8_t* A, const int8_t* B, float* C, const float* D, const float* bias_pointer, const float* scale_pointer, const uint32_t* mask_ptr, const int* mask_argsort_ptr, const int* indice_ptr, uint32_t* mask_out_ptr, uint32_t mask_filter, bool reverse_mask, float alpha = float(1), float beta = float(0), float act_alpha = float(0), float act_beta = float(0), tv::gemm::Activation act_type = tv::gemm::Activation::kNone, int split_k_slices = 1, bool d_is_bias = false);
};
} // namespace cpTuring_s8s8f32s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib