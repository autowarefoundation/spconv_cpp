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
std::tuple<int, int, int> GemmTunerSimple::extract_mnk(tv::TensorShape a_shape, tv::TensorShape b_shape, bool trans_a, bool trans_b, bool trans_c, std::tuple<int, int> arch, int shuffle_type, tv::TensorShape a_inds_shape, tv::TensorShape b_inds_shape, tv::TensorShape c_inds_shape, int hint)   {
  
  std::vector<int64_t> a_shape_vec(a_shape.begin(), a_shape.end());
  std::vector<int64_t> b_shape_vec(b_shape.begin(), b_shape.end());
  std::vector<int64_t> a_inds_shape_vec(a_inds_shape.begin(), a_inds_shape.end());
  std::vector<int64_t> b_inds_shape_vec(b_inds_shape.begin(), b_inds_shape.end());
  std::vector<int64_t> c_inds_shape_vec(c_inds_shape.begin(), c_inds_shape.end());
  return GemmMain::extract_mnk(a_shape_vec, b_shape_vec, trans_a,
                              trans_b, trans_c,
                              shuffle_type,
                              a_inds_shape_vec, b_inds_shape_vec,
                              c_inds_shape_vec);
}
} // namespace gemmops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib