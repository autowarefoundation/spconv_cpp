#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
void StaticAllocator::set_new_tensor_dict(std::unordered_map<std::string, tv::Tensor> tensor_dict)   {
  
  tensor_dict_ = tensor_dict;
  std::stringstream ss;
  for (auto& p : tensor_dict){
      tv::sstream_print(ss, p.first, p.second.shape(), tv::dtype_str(p.second.dtype()), "\n");
  }
  repr_ = ss.str();
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib