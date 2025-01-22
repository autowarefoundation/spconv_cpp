#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
tv::Tensor StaticAllocator::_get_raw_and_check(std::string name, std::vector<int64_t> shape, int dtype, int device, bool is_temp_memory)   {
  
  auto res = get_tensor_by_name(name);
  size_t total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  TV_ASSERT_RT_ERR(res.nbytes() >= total * tv::bit_size(tv::DType(dtype)) / 8 
      && res.device() == device, "alloc failed, tensor size too small", shape, res.shape());
  // if (is_temp_memory){
  // }else{
  //     // size must exactly match
  //     TV_ASSERT_RT_ERR(res.nbytes() == total * tv::bit_size(tv::DType(dtype)) / 8 
  //         && res.device() == device, "alloc failed, named memory size must match", shape, res.shape());
  // }
  return tv::from_blob(res.raw_data(), shape, tv::DType(dtype), device);
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib