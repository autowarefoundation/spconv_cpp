#include <spconvlib/spconv/csrc/sparse/maxpool/IndiceMaxPoolCPU.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace maxpool {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmDTypes = spconvlib::cumm::common::GemmDTypes;
void IndiceMaxPoolCPU::global_pool_rearrange(tv::Tensor out_indices, tv::Tensor coords, tv::Tensor counts)   {
  
  auto nhot = coords.dim(0);
  auto out_ptr = out_indices.data_ptr<int>();
  auto coord_ptr = coords.data_ptr<const int>();
  auto count_ptr = counts.data_ptr<int>();
  int indices_stride = coords.stride(0);
  for (int i = 0; i < nhot; ++i){
      int batch_idx = coord_ptr[0];
      if (batch_idx >= 0){
          out_ptr[batch_idx * nhot + (count_ptr[batch_idx]++)] = i;
      }
      coord_ptr += indices_stride;
  }
}
} // namespace maxpool
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib