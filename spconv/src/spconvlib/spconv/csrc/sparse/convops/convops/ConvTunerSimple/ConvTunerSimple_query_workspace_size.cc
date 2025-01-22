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
int ConvTunerSimple::query_workspace_size(tv::gemm::ConvAlgoDesp desp, int splitk, int op_type, int N, int C, int K, int kv)   {
  
  auto mnk = ConvMain::extract_mnk(op_type, N, C, K, kv, -1, -1, true);
  return desp.query_conv_workspace_size(
      std::get<0>(mnk), std::get<1>(mnk), std::get<2>(mnk),
      splitk, kv);
}
} // namespace convops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib