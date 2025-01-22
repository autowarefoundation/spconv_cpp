#pragma once
#include <limits>
#include <spconvlib/cumm/common/TensorView.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace utils {
namespace launch {
using TensorView = spconvlib::cumm::common::TensorView;
struct LaunchUtils {
  static constexpr int kMaxGridYZDim = 65535;
  /**
   * @param nhot 
   * @param num_features 
   */
  static std::tuple<int, int, int, int> get_blocks_threads_of_2d_tensor(int64_t nhot, int64_t num_features);
};
} // namespace launch
} // namespace utils
} // namespace csrc
} // namespace spconv
} // namespace spconvlib