#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
std::vector<int> GemmMainUnitTest::simple_select_tile_shape(int m, int n, int k, std::vector<int> tile_ms, std::vector<int> tile_ns, std::vector<int> tile_ks, std::unordered_map<int64_t, std::vector<int>> tile_shape_to_algos, bool large_k_first)   {
  
  auto iter_m_target = std::lower_bound(tile_ms.begin(), tile_ms.end(), m);
  auto iter_n_target = std::lower_bound(tile_ns.begin(), tile_ns.end(), n);
  auto iter_k_target = std::lower_bound(tile_ks.begin(), tile_ks.end(), k);
  if (iter_m_target == tile_ms.end()){
      iter_m_target = tile_ms.end() - 1;
  }
  if (iter_n_target == tile_ns.end()){
      iter_n_target = tile_ns.end() - 1;
  }
  if (iter_k_target == tile_ks.end()){
      iter_k_target = tile_ks.end() - 1;
  }
  // tv::ssprint(*iter_m_target, *iter_n_target, *iter_k_target);
  // try to find a valid configuration
  if (large_k_first){
      for (auto iter_k = iter_k_target; iter_k != tile_ks.begin() - 1; --iter_k){
          for (auto iter_n = iter_n_target; iter_n != tile_ns.begin() - 1; --iter_n){
              for (auto iter_m = iter_m_target; iter_m != tile_ms.begin() - 1; --iter_m){
                  int64_t tm = *iter_m;
                  int64_t tn = *iter_n;
                  int64_t tk = *iter_k;
                  int64_t tile_key = tm | (tn << 20) | (tk << 40);
                  auto target_iter = tile_shape_to_algos.find(tile_key);
                  if (target_iter != tile_shape_to_algos.end()){
                      return target_iter->second;
                  }
              }
          }
      }
  }
  else{
      for (auto iter_m = iter_m_target; iter_m != tile_ms.begin() - 1; --iter_m){
          for (auto iter_n = iter_n_target; iter_n != tile_ns.begin() - 1; --iter_n){
              for (auto iter_k = iter_k_target; iter_k != tile_ks.begin() - 1; --iter_k){
                  int64_t tm = *iter_m;
                  int64_t tn = *iter_n;
                  int64_t tk = *iter_k;
                  int64_t tile_key = tm | (tn << 20) | (tk << 40);
                  auto target_iter = tile_shape_to_algos.find(tile_key);
                  if (target_iter != tile_shape_to_algos.end()){
                      return target_iter->second;
                  }
              }
          }
      }
  }
  return {};
}
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib