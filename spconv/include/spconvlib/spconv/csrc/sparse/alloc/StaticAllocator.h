#pragma once
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/spconv/csrc/sparse/alloc/ExternalAllocator.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace alloc {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
struct StaticAllocator : public ExternalAllocator {
  std::unordered_map<std::string, tv::Tensor> tensor_dict_;
  std::string repr_;
  tv::Tensor thrust_tmp_tensor_;
  /**
   * @param tensor_dict 
   */
   StaticAllocator(std::unordered_map<std::string, tv::Tensor> tensor_dict);
  /**
   * @param tensor_dict 
   */
  void set_new_tensor_dict(std::unordered_map<std::string, tv::Tensor> tensor_dict);
  /**
   * @param name 
   * @param shape 
   * @param dtype 
   * @param device 
   * @param is_temp_memory 
   */
  virtual tv::Tensor _get_raw_and_check(std::string name, std::vector<int64_t> shape, int dtype, int device, bool is_temp_memory = false);
  /**
   * @param name 
   * @param shape 
   * @param dtype 
   * @param device 
   * @param stream 
   * @param is_temp_memory 
   * @param scale 
   */
  virtual tv::Tensor zeros(std::string name, std::vector<int64_t> shape, int dtype, int device, std::uintptr_t stream = 0, bool is_temp_memory = false, float scale = 1.0);
  /**
   * @param name 
   * @param shape 
   * @param dtype 
   * @param device 
   * @param stream 
   * @param is_temp_memory 
   * @param scale 
   */
  virtual tv::Tensor empty(std::string name, std::vector<int64_t> shape, int dtype, int device, std::uintptr_t stream = 0, bool is_temp_memory = false, float scale = 1.0);
  /**
   * @param name 
   * @param shape 
   * @param value 
   * @param dtype 
   * @param device 
   * @param stream 
   * @param is_temp_memory 
   */
  tv::Tensor full_int(std::string name, std::vector<int64_t> shape, int value, int dtype, int device, std::uintptr_t stream = 0, bool is_temp_memory = false);
  /**
   * @param name 
   * @param shape 
   * @param value 
   * @param dtype 
   * @param device 
   * @param stream 
   * @param is_temp_memory 
   */
  tv::Tensor full_float(std::string name, std::vector<int64_t> shape, float value, int dtype, int device, std::uintptr_t stream = 0, bool is_temp_memory = false);
  /**
   * @param name 
   */
  virtual tv::Tensor get_tensor_by_name(std::string name);
  /**
   * @param ten 
   */
  virtual void free(tv::Tensor ten);
  /**
   * @param ten 
   */
  virtual void free_noexcept(tv::Tensor ten);
};
} // namespace alloc
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib