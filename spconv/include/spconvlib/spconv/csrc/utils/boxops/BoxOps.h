#pragma once
#include <spconvlib/cumm/common/TensorView.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace utils {
namespace boxops {
using TensorView = spconvlib::cumm::common::TensorView;
struct BoxOps {
  
  static bool has_boost();
  /**
   * @param boxes 
   * @param order 
   * @param thresh 
   * @param eps 
   */
  static std::vector<int> non_max_suppression_cpu(tv::Tensor boxes, tv::Tensor order, float thresh, float eps = 0);
};
} // namespace boxops
} // namespace utils
} // namespace csrc
} // namespace spconv
} // namespace spconvlib