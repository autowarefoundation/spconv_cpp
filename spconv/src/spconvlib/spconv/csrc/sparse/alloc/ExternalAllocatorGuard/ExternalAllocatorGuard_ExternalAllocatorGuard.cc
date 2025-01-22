#include <spconvlib/spconv/csrc/sparse/alloc/ExternalAllocatorGuard.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
 ExternalAllocatorGuard::ExternalAllocatorGuard(tv::Tensor ten, std::function<void(tv::Tensor)> free_func) : tensor(ten), free_func(free_func)  {
  
}
 ExternalAllocatorGuard::ExternalAllocatorGuard()   {
  
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib