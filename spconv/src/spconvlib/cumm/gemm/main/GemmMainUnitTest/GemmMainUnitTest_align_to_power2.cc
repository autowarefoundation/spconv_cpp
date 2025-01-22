#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
int GemmMainUnitTest::align_to_power2(int val)   {
  
  size_t r = 0;
  size_t num_1_bit = val & 1 ? 1 : 0;
  while (val >>= 1) {
      r++;
      if (val & 1) {
          ++num_1_bit;
      }
  }
  if (num_1_bit == 1) {
      return 1 << r;
  } else {
      return 1 << (r + 1);
  }
}
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib