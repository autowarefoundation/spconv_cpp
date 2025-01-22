#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
tv::Tensor StaticAllocator::zeros(std::string name, std::vector<int64_t> shape, int dtype, int device, std::uintptr_t stream, bool is_temp_memory, float scale)   {
  
  auto tvctx = tv::Context();
  tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream));
  auto blob = _get_raw_and_check(name, shape, dtype, device, is_temp_memory);
  return blob.zero_(tvctx);
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib