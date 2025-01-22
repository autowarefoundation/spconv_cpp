#include <spconvlib/spconv/csrc/sparse/convops/convops/ConvTunerSimple.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
namespace convops {
using static_key_t = std::tuple<int, int, int, int, int, int, int, int, int, int>;
using algo_cache_key_t = std::tuple<int, int, int, int, int, int, int, int, bool>;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using ConvTuneResult = spconvlib::spconv::csrc::sparse::convops::ConvTuneResult;
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using CompileInfo = spconvlib::cumm::common::CompileInfo;
using ConvMain = spconvlib::cumm::conv::main::ConvMainUnitTest;
tv::gemm::NVRTCParams ConvTunerSimple::cached_get_nvrtc_params(tv::gemm::ConvAlgoDesp desp, std::tuple<int, int> arch, std::uintptr_t stream_int)   {
  
  TV_THROW_RT_ERR("not implemented in c++, must be overrided in python!!!");
}
} // namespace convops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib