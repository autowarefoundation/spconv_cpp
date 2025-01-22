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
 ConvTunerSimple::ConvTunerSimple(std::vector<tv::gemm::ConvAlgoDesp> desps) : desps_(desps)  {
  
  for (auto& d : desps){
      static_key_t static_key = std::make_tuple(
          int(d.layout_i), int(d.layout_w), int(d.layout_o),
          d.interleave_i, d.interleave_w, d.interleave_o, d.dtype_input(),
          d.dtype_weight(), d.dtype_output(), int(d.op_type));
      auto& vec = static_key_to_desps_[static_key];
      vec.push_back(d);
  }
  for (auto desp : ConvMain::get_all_conv_algo_desp()){
      prebuilt_names_.insert(desp.__repr__());
  }
}
} // namespace convops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib