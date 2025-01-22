#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
void GemmMainUnitTest::device_synchronize()   {
  
  checkCudaErrors(cudaDeviceSynchronize());
}
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib