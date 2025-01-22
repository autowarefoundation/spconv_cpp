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
std::tuple<int, int, int> GemmTunerSimple::extract_mnk_vector(std::vector<int64_t> a_shape, std::vector<int64_t> b_shape, bool trans_a, bool trans_b, bool trans_c, int shuffle_type, std::vector<int64_t> a_inds_shape, std::vector<int64_t> b_inds_shape, std::vector<int64_t> c_inds_shape)   {
  
  return GemmMain::extract_mnk(a_shape, b_shape, trans_a,
                              trans_b, trans_c,
                              shuffle_type,
                              a_inds_shape, b_inds_shape,
                              c_inds_shape);
}
} // namespace gemmops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib