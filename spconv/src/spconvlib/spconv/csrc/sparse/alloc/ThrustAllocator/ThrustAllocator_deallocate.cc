#include <spconvlib/spconv/csrc/sparse/alloc/ThrustAllocator.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using value_type = char;
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
void ThrustAllocator::deallocate(char * ptr, size_t num_bytes)   {
  
  return allocator_.free_noexcept(tv::from_blob(ptr, {int64_t(num_bytes)}, tv::uint8, 0));
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib