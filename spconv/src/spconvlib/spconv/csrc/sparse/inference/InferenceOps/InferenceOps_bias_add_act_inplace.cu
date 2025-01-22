#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>
#include <spconvlib/spconv/csrc/sparse/inference/ker/InferenceOpsKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace inference {
using TensorView = spconvlib::cumm::common::TensorView;
using LaunchUtils = spconvlib::spconv::csrc::utils::launch::LaunchUtils;
void InferenceOps::bias_add_act_inplace(tv::Tensor out, tv::Tensor bias, tv::gemm::Activation act_type, float alpha, float beta, std::uintptr_t stream)   {
  
  auto nhot = out.dim(0);
  auto cudastream = reinterpret_cast<cudaStream_t>(stream);
  TV_ASSERT_RT_ERR(bias.dim(0) == out.dim(1), "error");
  tv::dispatch<float, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){
      using T = TV_DECLTYPE(I);
      auto launchdims = LaunchUtils::get_blocks_threads_of_2d_tensor(nhot, out.dim(1));
      int num_blocks_X = std::get<0>(launchdims);
      int num_blocks_Y = std::get<1>(launchdims);
      dim3 blocks;
      dim3 threads(std::get<2>(launchdims), std::get<3>(launchdims));
      if (num_blocks_Y > kMaxGridYZDim){
          blocks = dim3(num_blocks_X * num_blocks_Y);
      }else{
          blocks = dim3(num_blocks_X, num_blocks_Y);
      }
      tv::cuda::Launch launcher = tv::cuda::Launch(blocks, threads, cudastream);
      tv::dispatch_int<0, 1>(int(num_blocks_Y > kMaxGridYZDim), [&](auto I2){
          constexpr bool OneDim = TV_DECLTYPE(I2)::value == 1;
          if (act_type == tv::gemm::Activation::kNone){
              launcher(ker::bias_add_inplace_kernel<T, OneDim>, out.data_ptr<T>(), bias.data_ptr<const T>(),
                  nhot, out.dim(1), num_blocks_X, num_blocks_Y);
          }else{
              launcher(ker::bias_add_act_inplace_kernel<T, OneDim>, out.data_ptr<T>(), bias.data_ptr<const T>(),
                  act_type, T(alpha), T(beta), nhot, out.dim(1), num_blocks_X, num_blocks_Y);
          }
      });
      TV_CHECK_CUDA_ERR_V2("bias add act failed!!!");
  });
}
} // namespace inference
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib