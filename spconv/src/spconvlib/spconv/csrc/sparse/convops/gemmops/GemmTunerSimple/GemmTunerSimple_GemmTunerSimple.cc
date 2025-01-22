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
 GemmTunerSimple::GemmTunerSimple(std::vector<tv::gemm::GemmAlgoDesp> desps) : desps_(desps)  {
  
  for (auto& d : desps){
      static_key_t static_key = std::make_tuple(d.trans_a(), d.trans_b(), d.trans_c(), d.dtype_a, d.dtype_b,
          d.dtype_c, int(d.shuffle_type));
      auto& vec = static_key_to_desps_[static_key];
      vec.push_back(d);
  }
  for (auto desp : GemmMain::get_all_algo_desp()){
      prebuilt_names_.insert(desp.__repr__());
  }
}
} // namespace gemmops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib