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
std::tuple<GemmTuneResult, bool> GemmTunerSimple::get_tuned_algo(int a_dtype, int b_dtype, int c_dtype, std::vector<int64_t> a_shape, std::vector<int64_t> b_shape, std::vector<int64_t> c_shape, bool trans_a, bool trans_b, bool trans_c, std::tuple<int, int> arch, int shuffle_type, std::vector<int64_t> a_inds_shape, std::vector<int64_t> b_inds_shape, std::vector<int64_t> c_inds_shape, int hint)   {
  
  auto mnk = GemmMain::extract_mnk(a_shape, b_shape, trans_a,
                              trans_b, trans_c,
                              shuffle_type,
                              a_inds_shape, b_inds_shape,
                              c_inds_shape);
  auto m = std::get<0>(mnk);
  auto n = std::get<1>(mnk);
  auto k = std::get<2>(mnk);
  GemmTuneResult res;
  bool exists = false;
  {
      std::lock_guard<std::mutex> guard(mutex_);
      algo_cache_key_t key;
      if (hint & 4){
          key = std::make_tuple(int(a_dtype), int(b_dtype), int(c_dtype), m, n);
          if (mn_cache_.find(key) != mn_cache_.end()){
              res = mn_cache_.at(key);
              exists = true;
          }
      }
      else if (hint & 2){
          key = std::make_tuple(int(a_dtype), int(b_dtype), int(c_dtype), n, k);
          if (nk_dgrad_cache_.find(key) != nk_dgrad_cache_.end()){
              res = nk_dgrad_cache_.at(key);
              exists = true;
          }
      }
      else if (hint & 1){
          key = std::make_tuple(int(a_dtype), int(b_dtype), int(c_dtype), n, k);
          if (nk_forward_cache_.find(key) != nk_forward_cache_.end()){
              res = nk_forward_cache_.at(key);
              exists = true;
          }
      }
      else{
          TV_THROW_RT_ERR("not implemented");
      }
  }
  return std::make_tuple(res, exists);
}
} // namespace gemmops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib