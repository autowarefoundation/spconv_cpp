#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
void GemmMainUnitTest::matmul2(tv::gemm::GemmParams params)   {
  
  auto& algo_desp = params.algo_desp;
  if (algo_desp.algo == "Simt"&& static_cast<int>(algo_desp.shuffle_type) == 1){
    if (algo_desp.dtype_a == tv::DType(0) && algo_desp.dtype_b == tv::DType(0) && algo_desp.dtype_c == tv::DType(0)){
      return matmul_split_Simt_f32f32f32_1(params);
    }
    if (algo_desp.dtype_a == tv::DType(7) && algo_desp.dtype_b == tv::DType(7) && algo_desp.dtype_c == tv::DType(7)){
      return matmul_split_Simt_f16f16f16_1(params);
    }
  }
  if (algo_desp.algo == "Volta"&& static_cast<int>(algo_desp.shuffle_type) == 1){
    if (algo_desp.dtype_a == tv::DType(7) && algo_desp.dtype_b == tv::DType(7) && algo_desp.dtype_c == tv::DType(7)){
      return matmul_split_Volta_f16f16f16_1(params);
    }
  }
  if (algo_desp.algo == "Turing"&& static_cast<int>(algo_desp.shuffle_type) == 1){
    if (algo_desp.dtype_a == tv::DType(7) && algo_desp.dtype_b == tv::DType(7) && algo_desp.dtype_c == tv::DType(7)){
      return matmul_split_Turing_f16f16f16_1(params);
    }
  }
  TV_THROW_RT_ERR("can't find any suitable algo for your parameters.");
}
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib