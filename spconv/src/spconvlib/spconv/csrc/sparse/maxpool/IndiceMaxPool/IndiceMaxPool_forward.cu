#include <spconvlib/spconv/csrc/sparse/maxpool/IndiceMaxPool.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace maxpool {
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using LaunchUtils = spconvlib::spconv::csrc::utils::launch::LaunchUtils;
void IndiceMaxPool::forward(tv::Tensor out, tv::Tensor in, tv::Tensor out_inds, tv::Tensor in_inds, std::uintptr_t stream)   {
  
  auto nhot = out_inds.dim(0);
  auto cudastream = reinterpret_cast<cudaStream_t>(stream);
  tv::dispatch<float, double, tv::half_t, tv::bfloat16_t, int8_t>(out.dtype(), [&](auto I){
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
          launcher(forward_kernel<T, OneDim>, out.data_ptr<T>(), in.data_ptr<const T>(),
              out_inds.data_ptr<const int>(), in_inds.data_ptr<const int>(), nhot, out.dim(1),
              num_blocks_X, num_blocks_Y);
      });
      TV_CHECK_CUDA_ERR_V2("max pool fwd failed!!!");
  });
}
} // namespace maxpool
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib