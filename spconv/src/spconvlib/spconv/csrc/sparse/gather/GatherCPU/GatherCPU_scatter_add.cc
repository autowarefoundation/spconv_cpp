#include <spconvlib/spconv/csrc/sparse/gather/GatherCPU.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace gather {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmDTypes = spconvlib::cumm::common::GemmDTypes;
void GatherCPU::scatter_add(tv::Tensor out, tv::Tensor in, tv::Tensor inds)   {
  
  // tv::check_shape(inds, {in.dim(0)});
  auto nhot = inds.dim(0);
  int channel = in.dim(1);
  tv::dispatch<float, double, tv::bfloat16_t, tv::half_t>(out.dtype(), [&](auto I){
      using T = TV_DECLTYPE(I);
      auto indices_data = inds.data_ptr<const int>();
      const T *buffer_data = in.data_ptr<const T>();
      T *features_data = out.data_ptr<T>();
      const T *buf = in.data_ptr<const T>();
      T *out_ptr = out.data_ptr<T>();
      tv::kernel_1d(out.device(), nhot, [&](int begin, int end, int step){
          for (int i = begin; i < end; i += step) {
              buf = buffer_data + i * channel;
              out_ptr = features_data + indices_data[i] * channel;
              for (int j = 0; j < channel; ++j) {
                  out_ptr[j] = out_ptr[j] + buf[j];
              }
          }
      });
  });
}
} // namespace gather
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib