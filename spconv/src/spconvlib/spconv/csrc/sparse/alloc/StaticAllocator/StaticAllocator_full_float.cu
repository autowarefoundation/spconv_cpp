#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/alloc/cudakers/CudaCommonKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
tv::Tensor StaticAllocator::full_float(std::string name, std::vector<int64_t> shape, float value, int dtype, int device, std::uintptr_t stream, bool is_temp_memory)   {
  
  auto tvctx = tv::Context();
  auto blob = _get_raw_and_check(name, shape, dtype, device, is_temp_memory);
  tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream));
  using dtypes_t = std::tuple<float, double>;
  tv::Dispatch<dtypes_t>()(blob.dtype(), [&](auto I){
      using T = TV_DECLTYPE(I);
      tv::cuda::Launch lanucher_fill(blob.size(), reinterpret_cast<cudaStream_t>(stream));
      lanucher_fill(cudakers::fill_kernel<T>, blob.data_ptr<T>(), value, blob.size());
  });
  return blob;
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib