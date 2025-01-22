#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
tv::Tensor StaticAllocator::get_tensor_by_name(std::string name)   {
  
  TV_ASSERT_RT_ERR(tensor_dict_.find(name) != tensor_dict_.end(), "can't find", name, "exists:\n", repr_);
  return tensor_dict_.at(name);
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib