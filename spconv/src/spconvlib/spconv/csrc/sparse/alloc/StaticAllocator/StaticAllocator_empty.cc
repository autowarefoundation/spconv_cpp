#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
tv::Tensor StaticAllocator::empty(std::string name, std::vector<int64_t> shape, int dtype, int device, std::uintptr_t stream, bool is_temp_memory, float scale)   {
  
  if (name == "ThrustTemp"){
      // thrust tmp shouldn't inside tensor_dict. use a simple method to allocate
      // we assume each allocator always handle one stream
      // so we can just use one tensor
      tv::Tensor res = thrust_tmp_tensor_;
      if (res.empty()){
          res = tv::empty(shape, tv::DType(dtype), device);
          thrust_tmp_tensor_ = res;
      }
      if (shape[0] > thrust_tmp_tensor_.dim(0)){
          res = tv::empty({int64_t(shape[0] * 1.5)}, tv::DType(dtype), device);
          thrust_tmp_tensor_ = res;
      }
      return res;
  }else{
      auto blob = _get_raw_and_check(name, shape, dtype, device, is_temp_memory);
      return blob;
  }
}
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib