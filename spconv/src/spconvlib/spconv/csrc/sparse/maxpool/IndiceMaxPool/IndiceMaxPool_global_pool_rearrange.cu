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
void IndiceMaxPool::global_pool_rearrange(tv::Tensor out_indices, tv::Tensor coords, tv::Tensor counts, std::uintptr_t stream)   {
  
  auto nhot = coords.dim(0);
  auto cudastream = reinterpret_cast<cudaStream_t>(stream);
  tv::cuda::Launch launcher = tv::cuda::Launch(nhot, cudastream);
  launcher(global_pool_rearrange_kernel, out_indices.data_ptr<int>(), 
      coords.data_ptr<const int>(), counts.data_ptr<int>(), nhot, 
      coords.stride(0));
  TV_CHECK_CUDA_ERR_V2("global_pool_feature_rearrange failed!!!");
}
} // namespace maxpool
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib