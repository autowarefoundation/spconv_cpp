#pragma once
#include <spconvlib/cumm/common/TensorViewMath.h>
#include <spconvlib/cumm/conv/main/cpSimt_f32f32f32f32f32tnt_m64n64k8m32n32k8A1_200_C301LLL_SK/cp/ConvProblem.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace cpSimt_f32f32f32f32f32tnt_m64n64k8m32n32k8A1_200_C301LLL_SK {
namespace itera_p {
using TensorViewMath = spconvlib::cumm::common::TensorViewMath;
using ConvProblem = spconvlib::cumm::conv::main::cpSimt_f32f32f32f32f32tnt_m64n64k8m32n32k8A1_200_C301LLL_SK::cp::ConvProblem;
struct SparseParams {
  int filter_c_delta;
  int inc_c_next;
  int inc_c_reset;
  int const* indice_ptr_;
  int const* mask_argsort_ptr_;
  int RS;
  __forceinline__ __host__ __device__  SparseParams(ConvProblem const& problem, int const* indice_ptr, int const* mask_argsort_ptr) : indice_ptr_(indice_ptr), mask_argsort_ptr_(mask_argsort_ptr)  {
    
    RS = problem.kernel_volume;
    filter_c_delta = 8 * problem.split_k_slices;
    inc_c_next = filter_c_delta * 4 ;
  }
  __forceinline__ __host__ __device__ void set_inc_reset_for_inc_k_first(int gemm_iters_k)   {
    
    inc_c_reset = (- filter_c_delta) * gemm_iters_k * 4  ;
  }
};
} // namespace itera_p
} // namespace cpSimt_f32f32f32f32f32tnt_m64n64k8m32n32k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib