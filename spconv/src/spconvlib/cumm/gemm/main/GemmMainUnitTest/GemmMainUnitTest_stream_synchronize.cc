#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
void GemmMainUnitTest::stream_synchronize(std::uintptr_t stream)   {
  
  auto res = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  if (res){
      TV_THROW_RT_ERR("CUDA error", int(res));
  }
}
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib