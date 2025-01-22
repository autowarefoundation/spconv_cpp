#pragma once
#include <functional>
#include <memory>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/spconv/csrc/sparse/alloc/ExternalAllocator.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
struct ThrustAllocator {
  using value_type = char;
  ExternalAllocator& allocator_;
  /**
   * @param allocator 
   */
   ThrustAllocator(ExternalAllocator& allocator);
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
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib