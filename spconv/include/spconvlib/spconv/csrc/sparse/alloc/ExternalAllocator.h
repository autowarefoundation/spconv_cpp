#pragma once
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/spconv/csrc/sparse/alloc/ExternalAllocatorGuard.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocatorGuard = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocatorGuard;
struct ExternalAllocator {
  using guard_t = std::shared_ptr<ExternalAllocatorGuard>;
  /**
   * @param name 
   * @param shape 
   * @param dtype 
   * @param device 
   * @param stream 
   * @param is_temp_memory 
   * @param scale 
   */
  virtual tv::Tensor zeros(std::string name, std::vector<int64_t> shape, int dtype, int device, std::uintptr_t stream = 0, bool is_temp_memory = false, float scale = 1.0) = 0;
  /**
   * @param name 
   * @param shape 
   * @param dtype 
   * @param device 
   * @param stream 
   * @param is_temp_memory 
   * @param scale 
   */
  virtual tv::Tensor empty(std::string name, std::vector<int64_t> shape, int dtype, int device, std::uintptr_t stream = 0, bool is_temp_memory = false, float scale = 1.0) = 0;
  /**
   * @param name 
   * @param shape 
   * @param value 
   * @param dtype 
   * @param device 
   * @param stream 
   * @param is_temp_memory 
   */
  virtual tv::Tensor full_int(std::string name, std::vector<int64_t> shape, int value, int dtype, int device, std::uintptr_t stream = 0, bool is_temp_memory = false) = 0;
  /**
   * @param name 
   * @param shape 
   * @param value 
   * @param dtype 
   * @param device 
   * @param stream 
   * @param is_temp_memory 
   */
  virtual tv::Tensor full_float(std::string name, std::vector<int64_t> shape, float value, int dtype, int device, std::uintptr_t stream = 0, bool is_temp_memory = false) = 0;
  /**
   * @param name 
   */
  virtual tv::Tensor get_tensor_by_name(std::string name) = 0;
  /**
   * @param ten 
   */
  virtual void free(tv::Tensor ten) = 0;
  /**
   * @param ten 
   */
  virtual void free_noexcept(tv::Tensor ten) = 0;
  /**
   * @param shape 
   * @param dtype 
   * @param device 
   * @param name 
   * @param stream 
   */
  std::shared_ptr<ExternalAllocatorGuard> zeros_guard(std::vector<int64_t> shape, int dtype, int device, std::string name = "", std::uintptr_t stream = 0);
  /**
   * @param shape 
   * @param dtype 
   * @param device 
   * @param name 
   * @param stream 
   */
  std::shared_ptr<ExternalAllocatorGuard> empty_guard(std::vector<int64_t> shape, int dtype, int device, std::string name = "", std::uintptr_t stream = 0);
  /**
   * @param shape 
   * @param value 
   * @param dtype 
   * @param device 
   * @param name 
   * @param stream 
   */
  std::shared_ptr<ExternalAllocatorGuard> full_int_guard(std::vector<int64_t> shape, int value, int dtype, int device, std::string name = "", std::uintptr_t stream = 0);
  /**
   * @param shape 
   * @param value 
   * @param dtype 
   * @param device 
   * @param name 
   * @param stream 
   */
  std::shared_ptr<ExternalAllocatorGuard> full_float_guard(std::vector<int64_t> shape, int value, int dtype, int device, std::string name = "", std::uintptr_t stream = 0);
};
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib