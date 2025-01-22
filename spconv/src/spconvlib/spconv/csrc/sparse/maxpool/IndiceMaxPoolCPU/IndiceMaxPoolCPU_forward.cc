#include <spconvlib/spconv/csrc/sparse/maxpool/IndiceMaxPoolCPU.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace maxpool {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmDTypes = spconvlib::cumm::common::GemmDTypes;
void IndiceMaxPoolCPU::forward(tv::Tensor out, tv::Tensor in, tv::Tensor out_inds, tv::Tensor in_inds, std::uintptr_t stream)   {
  
  int nhot = out_inds.dim(0);
  int num_features = in.dim(1);
  tv::dispatch<float, double, tv::half_t, tv::bfloat16_t>(out.dtype(), [&](auto I){
      using T = TV_DECLTYPE(I);
      auto out_features = out.data_ptr<T>();
      auto in_features = in.data_ptr<const T>();
      auto in_indices = in_inds.data_ptr<const int>();
      auto out_indices = out_inds.data_ptr<const int>();
      tv::kernel_1d(out.device(), nhot, [&](int begin, int end, int step){
          for (int i = begin; i < end; i += step) {
              int in_idx = in_indices[i];
              int out_idx = out_indices[i];
              auto in_ptr = in_features + in_idx * num_features;
              auto out_ptr = out_features + out_idx * num_features;
              for (int j = 0; j < num_features; ++j) {
                  auto in = in_ptr[j];
                  auto out = out_ptr[j];
                  if (in > out){
                      out_ptr[j] = in;
                  }
              }
          }
      });
  });
}
} // namespace maxpool
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib