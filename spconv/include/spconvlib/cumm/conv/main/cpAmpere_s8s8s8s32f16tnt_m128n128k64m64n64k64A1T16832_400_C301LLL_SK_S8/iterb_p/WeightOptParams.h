#pragma once
#include <spconvlib/cumm/common/TensorViewMath.h>
#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f16tnt_m128n128k64m64n64k64A1T16832_400_C301LLL_SK_S8/lb/TensorGeneric.h>
#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f16tnt_m128n128k64m64n64k64A1T16832_400_C301LLL_SK_S8/cp/ConvProblem.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace cpAmpere_s8s8s8s32f16tnt_m128n128k64m64n64k64A1T16832_400_C301LLL_SK_S8 {
namespace iterb_p {
using TensorViewMath = spconvlib::cumm::common::TensorViewMath;
using Layout = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f16tnt_m128n128k64m64n64k64A1T16832_400_C301LLL_SK_S8::lb::TensorGeneric;
using ConvProblem = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f16tnt_m128n128k64m64n64k64A1T16832_400_C301LLL_SK_S8::cp::ConvProblem;
struct WeightOptParams {
  Layout layout;
  int64_t inc_strided;
  int64_t inc_rs;
  int64_t inc_c;
  int filter_c_delta;
  int stride_rsc_bytes;
  int64_t inc_c_reset;
  __forceinline__ __host__ __device__  WeightOptParams(ConvProblem const& problem, Layout const& layout) : layout(layout)  {
    
    // int kernel_prod = problem.kernel_volume;
    filter_c_delta = 64 * problem.split_k_slices;
    inc_strided = int64_t(layout.strides[0]) * 8;
    stride_rsc_bytes = layout.strides[0] * 8 / 8;
    // back to strided start, then inc c
    inc_c = filter_c_delta - inc_strided * int64_t(3);
    inc_rs = int64_t(layout.strides[1]);
    // inc_c_reset = -gemm_iters_k * filter_c_delta * 8 / 8;
    inc_rs = inc_rs * 8 / 8;
    inc_strided = inc_strided * 8 / 8;
    inc_c = inc_c * 8 / 8;
  }
  __forceinline__ __host__ __device__ void set_inc_reset_for_inc_k_first(int gemm_iters_k = -1)   {
    
    inc_c_reset = -gemm_iters_k * filter_c_delta * 8 / 8;
  }
};
} // namespace iterb_p
} // namespace cpAmpere_s8s8s8s32f16tnt_m128n128k64m64n64k64A1T16832_400_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib