#include <spconvlib/cumm/conv/main/ConvMainUnitTest.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using ConvNVRTCParams = spconvlib::cumm::conv::kernel::ConvNVRTCParams;
using CummNVRTCLib = spconvlib::cumm::common::CummNVRTCLib;
std::array<int, 3> ConvMainUnitTest::extract_mnk(int op_type, int N, int C, int K, int kernel_volume, int in_prod, int out_prod, bool mask_sparse)   {
  
  auto op_type_enum = static_cast<tv::gemm::ConvOpType>(op_type);
  auto res = tv::gemm::implicit_gemm_mnk(op_type_enum, N, C, K, 
      kernel_volume, in_prod, out_prod, mask_sparse);
  return {res[0], res[1], res[2]};
}
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib