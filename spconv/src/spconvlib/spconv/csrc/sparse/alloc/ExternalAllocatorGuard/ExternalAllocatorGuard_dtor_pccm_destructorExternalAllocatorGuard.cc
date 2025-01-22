#include <spconvlib/spconv/csrc/sparse/alloc/ExternalAllocatorGuard.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
 ExternalAllocatorGuard::~ExternalAllocatorGuard()   {
  
  if (!tensor.empty() && free_func){
      free_func(tensor);
  }
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib