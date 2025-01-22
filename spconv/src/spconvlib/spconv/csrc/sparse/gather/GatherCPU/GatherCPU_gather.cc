#include <spconvlib/spconv/csrc/sparse/gather/GatherCPU.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace gather {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmDTypes = spconvlib::cumm::common::GemmDTypes;
void GatherCPU::gather(tv::Tensor out, tv::Tensor in, tv::Tensor inds)   {
  
  // tv::check_shape(inds, {out.dim(0)});
  auto nhot = inds.dim(0);
  int channel = in.dim(1);
  tv::dispatch<float, double, tv::bfloat16_t, tv::half_t>(out.dtype(), [&](auto I){
      auto indices_data = inds.data_ptr<const int>();
      using T = TV_DECLTYPE(I);
      T *buffer_data = out.data_ptr<T>();
      const T *features_data = in.data_ptr<const T>();
      tv::kernel_1d(out.device(), nhot, [&](int begin, int end, int step){
          for (int i = begin; i < end; i += step) {
              std::memcpy(buffer_data + i * channel,
                          features_data + indices_data[i] * channel,
                          sizeof(T) * channel);
          }
      });
  });
}
} // namespace gather
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib