#include <spconvlib/spconv/csrc/sparse/alloc/ExternalAllocator.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using guard_t = std::shared_ptr<ExternalAllocatorGuard>;
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocatorGuard = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocatorGuard;
std::shared_ptr<ExternalAllocatorGuard> ExternalAllocator::zeros_guard(std::vector<int64_t> shape, int dtype, int device, std::string name, std::uintptr_t stream)   {
  
  // "" means temp memory
  auto ten = zeros(name, shape, dtype, device, stream, true);
  return std::make_shared<ExternalAllocatorGuard>(ten, [this](tv::Tensor ten){
      this->free(ten);
  });
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib