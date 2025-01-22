#include <spconvlib/spconv/csrc/sparse/all/ThrustCustomAllocatorV2.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
using value_type = char;
using TensorView = spconvlib::cumm::common::TensorView;
char* ThrustCustomAllocatorV2::allocate(std::ptrdiff_t num_bytes)   {
  
  if (alloc_func){
      char* result = reinterpret_cast<char*>(alloc_func(num_bytes));
      return result;
  }
  else{
      TV_THROW_RT_ERR("set alloc function first.");
  }
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib