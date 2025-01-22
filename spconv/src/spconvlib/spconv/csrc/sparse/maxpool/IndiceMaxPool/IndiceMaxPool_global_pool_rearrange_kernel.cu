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
__global__ void global_pool_rearrange_kernel(int* out_indices, const int* coords, int* counts, int num_indices, int indices_stride)   {
  
  for (int i : tv::KernelLoopX<int>(num_indices)) {
      int batch_idx = coords[i * indices_stride];
      if (batch_idx >= 0){
          auto old = atomicAdd(counts + batch_idx, 1);
          out_indices[batch_idx * num_indices + old] = i;
      }
  }
}
} // namespace maxpool
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib