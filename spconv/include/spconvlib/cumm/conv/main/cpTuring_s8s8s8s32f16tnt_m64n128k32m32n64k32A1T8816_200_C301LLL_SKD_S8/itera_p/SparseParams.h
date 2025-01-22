#pragma once
#include <spconvlib/cumm/common/TensorViewMath.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8s8s32f16tnt_m64n128k32m32n64k32A1T8816_200_C301LLL_SKD_S8/cp/ConvProblem.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace cpTuring_s8s8s8s32f16tnt_m64n128k32m32n64k32A1T8816_200_C301LLL_SKD_S8 {
namespace itera_p {
using TensorViewMath = spconvlib::cumm::common::TensorViewMath;
using ConvProblem = spconvlib::cumm::conv::main::cpTuring_s8s8s8s32f16tnt_m64n128k32m32n64k32A1T8816_200_C301LLL_SKD_S8::cp::ConvProblem;
struct SparseParams {
  int filter_c_delta;
  int inc_c_next;
  int inc_c_reset;
  int const* indice_ptr_;
  int const* mask_argsort_ptr_;
  int RS;
  __forceinline__ __host__ __device__  SparseParams(ConvProblem const& problem, int const* indice_ptr, int const* mask_argsort_ptr) : indice_ptr_(indice_ptr), mask_argsort_ptr_(mask_argsort_ptr)  {
    
    RS = problem.kernel_volume;
    filter_c_delta = 32 * problem.split_k_slices;
    inc_c_next = filter_c_delta * 1 ;
  }
  __forceinline__ __host__ __device__ void set_inc_reset_for_inc_k_first(int gemm_iters_k)   {
    
    inc_c_reset = (- filter_c_delta) * gemm_iters_k * 1  ;
  }
};
} // namespace itera_p
} // namespace cpTuring_s8s8s8s32f16tnt_m64n128k32m32n64k32A1T8816_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib