#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>
#include <spconvlib/spconv/csrc/sparse/inference/ker/InferenceOpsKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace inference {
using TensorView = spconvlib::cumm::common::TensorView;
using LaunchUtils = spconvlib::spconv::csrc::utils::launch::LaunchUtils;
void InferenceOps::activation_inplace(tv::Tensor out, tv::gemm::Activation act_type, float alpha, float beta, std::uintptr_t stream)   {
  
  auto nhot = out.size();
  auto cudastream = reinterpret_cast<cudaStream_t>(stream);
  tv::cuda::Launch launcher = tv::cuda::Launch(nhot, cudastream);
  tv::dispatch<float, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){
      using T = TV_DECLTYPE(I);
      launcher(ker::activation_inplace_kernel<T>, out.data_ptr<T>(), act_type, T(alpha), T(beta),
          nhot);
      TV_CHECK_CUDA_ERR_V2("bias add act failed!!!");
  });
}
} // namespace inference
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib