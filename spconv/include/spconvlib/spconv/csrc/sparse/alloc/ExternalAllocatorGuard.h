#pragma once
#include <spconvlib/cumm/common/TensorView.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
struct ExternalAllocatorGuard {
  tv::Tensor tensor;
  std::function<void(tv::Tensor)> free_func;
  /**
   * @param ten 
   * @param free_func 
   */
   ExternalAllocatorGuard(tv::Tensor ten, std::function<void(tv::Tensor)> free_func);
  
   ExternalAllocatorGuard();
  
  virtual  ~ExternalAllocatorGuard();
};
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib