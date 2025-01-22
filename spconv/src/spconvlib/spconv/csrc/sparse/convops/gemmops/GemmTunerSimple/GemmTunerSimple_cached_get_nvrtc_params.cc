#include <spconvlib/spconv/csrc/sparse/convops/gemmops/GemmTunerSimple.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
namespace gemmops {
using static_key_t = std::tuple<bool, bool, bool, int, int, int, int>;
using algo_cache_key_t = std::tuple<int, int, int, int, int>;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using GemmTuneResult = spconvlib::spconv::csrc::sparse::convops::GemmTuneResult;
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using CompileInfo = spconvlib::cumm::common::CompileInfo;
using GemmMain = spconvlib::cumm::gemm::main::GemmMainUnitTest;
tv::gemm::NVRTCParams GemmTunerSimple::cached_get_nvrtc_params(tv::gemm::GemmAlgoDesp desp, std::tuple<int, int> arch, std::uintptr_t stream_int)   {
  
  TV_THROW_RT_ERR("not implemented in c++, must be overrided in python!!!");
}
} // namespace gemmops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib