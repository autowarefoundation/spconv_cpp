#include <spconvlib/spconv/csrc/sparse/alloc/ThrustAllocator.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using value_type = char;
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
 ThrustAllocator::ThrustAllocator(ExternalAllocator& allocator) : allocator_(allocator)  {
  
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib