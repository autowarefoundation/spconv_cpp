#include <spconvlib/cumm/conv/main/ConvMainUnitTest.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using ConvNVRTCParams = spconvlib::cumm::conv::kernel::ConvNVRTCParams;
using CummNVRTCLib = spconvlib::cumm::common::CummNVRTCLib;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
void ConvMainUnitTest::implicit_gemm2(tv::gemm::ConvParams params)   {
  
  auto& algo_desp = params.conv_algo_desp;
  if (algo_desp.algo == "Simt"&& static_cast<int>(algo_desp.shuffle_type) == 0){
    if (algo_desp.dtype_a == tv::DType(0) && algo_desp.dtype_b == tv::DType(0) && algo_desp.dtype_c == tv::DType(0)){
      return matmul_split_Simt_f32f32f32_0(params);
    }
    if (algo_desp.dtype_a == tv::DType(7) && algo_desp.dtype_b == tv::DType(7) && algo_desp.dtype_c == tv::DType(7)){
      return matmul_split_Simt_f16f16f16_0(params);
    }
  }
  if (algo_desp.algo == "Volta"&& static_cast<int>(algo_desp.shuffle_type) == 0){
    if (algo_desp.dtype_a == tv::DType(7) && algo_desp.dtype_b == tv::DType(7) && algo_desp.dtype_c == tv::DType(7)){
      return matmul_split_Volta_f16f16f16_0(params);
    }
  }
  if (algo_desp.algo == "Turing"&& static_cast<int>(algo_desp.shuffle_type) == 0){
    if (algo_desp.dtype_a == tv::DType(3) && algo_desp.dtype_b == tv::DType(3) && algo_desp.dtype_c == tv::DType(3)){
      return matmul_split_Turing_s8s8s8_0(params);
    }
    if (algo_desp.dtype_a == tv::DType(3) && algo_desp.dtype_b == tv::DType(3) && algo_desp.dtype_c == tv::DType(0)){
      return matmul_split_Turing_s8s8f32_0(params);
    }
    if (algo_desp.dtype_a == tv::DType(3) && algo_desp.dtype_b == tv::DType(3) && algo_desp.dtype_c == tv::DType(7)){
      return matmul_split_Turing_s8s8f16_0(params);
    }
    if (algo_desp.dtype_a == tv::DType(7) && algo_desp.dtype_b == tv::DType(7) && algo_desp.dtype_c == tv::DType(7)){
      return matmul_split_Turing_f16f16f16_0(params);
    }
  }
  if (algo_desp.algo == "Ampere"&& static_cast<int>(algo_desp.shuffle_type) == 0){
    if (algo_desp.dtype_a == tv::DType(0) && algo_desp.dtype_b == tv::DType(0) && algo_desp.dtype_c == tv::DType(0)){
      return matmul_split_Ampere_f32f32f32_0(params);
    }
    if (algo_desp.dtype_a == tv::DType(7) && algo_desp.dtype_b == tv::DType(7) && algo_desp.dtype_c == tv::DType(7)){
      return matmul_split_Ampere_f16f16f16_0(params);
    }
    if (algo_desp.dtype_a == tv::DType(3) && algo_desp.dtype_b == tv::DType(3) && algo_desp.dtype_c == tv::DType(3)){
      return matmul_split_Ampere_s8s8s8_0(params);
    }
    if (algo_desp.dtype_a == tv::DType(3) && algo_desp.dtype_b == tv::DType(3) && algo_desp.dtype_c == tv::DType(0)){
      return matmul_split_Ampere_s8s8f32_0(params);
    }
    if (algo_desp.dtype_a == tv::DType(3) && algo_desp.dtype_b == tv::DType(3) && algo_desp.dtype_c == tv::DType(7)){
      return matmul_split_Ampere_s8s8f16_0(params);
    }
  }
  TV_THROW_RT_ERR("can't find any suitable algo for your parameters.", 
      algo_desp.algo, algo_desp.dtype_a, algo_desp.dtype_b, algo_desp.dtype_c,
      algo_desp.__repr__());
}
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib