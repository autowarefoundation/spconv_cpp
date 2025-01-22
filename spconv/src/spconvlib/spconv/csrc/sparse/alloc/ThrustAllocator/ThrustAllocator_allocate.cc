#include <spconvlib/spconv/csrc/sparse/alloc/ThrustAllocator.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using value_type = char;
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
char* ThrustAllocator::allocate(std::ptrdiff_t num_bytes)   {
  
  auto ten = allocator_.empty("ThrustTemp", {num_bytes}, tv::uint8, 0);
  return reinterpret_cast<char*>(ten.raw_data());
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib