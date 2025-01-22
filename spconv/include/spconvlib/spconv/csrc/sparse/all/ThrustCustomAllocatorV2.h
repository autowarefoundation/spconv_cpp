#pragma once
#include <functional>
#include <memory>
#include <spconvlib/cumm/common/TensorView.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
using TensorView = spconvlib::cumm::common::TensorView;
struct ThrustCustomAllocatorV2 {
  using value_type = char;
  std::function<std::uintptr_t(std::size_t)> alloc_func;
  /**
   * @param num_bytes 
   */
  char* allocate(std::ptrdiff_t num_bytes);
  /**
   * @param ptr 
   * @param num_bytes 
   */
  void deallocate(char * ptr, size_t num_bytes);
};
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib