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
std::vector<std::string> ConvTunerSimple::get_available_algo_str_from_arch(std::tuple<int, int> arch)   {
  
  std::vector<std::string> res;
  auto arch_cur_4 = std::make_tuple(int(8), int(0));
  if (arch >= arch_cur_4){
    res.push_back("Ampere");
  }
  auto arch_cur_3 = std::make_tuple(int(7), int(5));
  if (arch >= arch_cur_3){
    res.push_back("Turing");
  }
  auto arch_cur_2 = std::make_tuple(int(7), int(0));
  if (arch >= arch_cur_2){
    res.push_back("Volta");
  }
  auto arch_cur_1 = std::make_tuple(int(6), int(1));
  if (arch >= arch_cur_1){
    res.push_back("SimtDP4A");
    res.push_back("SimtDP2A");
  }
  auto arch_cur_0 = std::make_tuple(int(3), int(5));
  if (arch >= arch_cur_0){
    res.push_back("Simt");
  }
  return res;
}
} // namespace convops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib