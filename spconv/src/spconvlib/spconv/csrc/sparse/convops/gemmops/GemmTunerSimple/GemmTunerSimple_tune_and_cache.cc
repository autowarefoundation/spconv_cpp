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
std::tuple<GemmTuneResult, float> GemmTunerSimple::tune_and_cache(tv::Tensor a, tv::Tensor b, tv::Tensor c, bool trans_a, bool trans_b, bool trans_c, std::tuple<int, int> arch, int shuffle_type, tv::Tensor a_inds, tv::Tensor b_inds, tv::Tensor c_inds, int hint, float alpha, float beta, std::uintptr_t stream_int, int num_run, bool use_tf32)   {
  
  TV_ASSERT_RT_ERR(num_run > 1, "error");
  auto mnk = extract_mnk(a.shape(), b.shape(), trans_a,
                              trans_b, trans_c,
                              arch,
                              shuffle_type,
                              a_inds.shape(), b_inds.shape(),
                              c_inds.shape());
  auto m = std::get<0>(mnk);
  auto n = std::get<1>(mnk);
  auto k = std::get<2>(mnk);
  auto avail = get_all_available(a, b, c, trans_a, trans_b, 
      trans_c, arch, shuffle_type, use_tf32);
  auto c_ = c.clone_whole_storage();
  std::vector<GemmTuneResult> all_profile_res;
  std::unordered_set<int> splitk_tests;
  std::vector<float> times;
  float min_time = -1;
  for (auto& desp : avail){
      tv::gemm::GemmParams params;
      if (desp.is_nvrtc || prebuilt_names_.find(desp.__repr__()) == prebuilt_names_.end()){
          params.nvrtc_params = cached_get_nvrtc_params(desp, arch, stream_int);
      }
      params.a = a;
      params.b = b;
      params.c = c_;
      params.a_inds = a_inds;
      params.b_inds = b_inds;
      params.c_inds = c_inds;
      params.algo_desp = desp;
      params.alpha = alpha;
      params.beta = beta;
      params.stream = stream_int;
      if (desp.split_k_serial() && (hint & 4)){
          splitk_tests = {1, 2, 4, 8, 16, 32, 64};
          splitk_tests.insert(int(a.dim(0)) / std::min(1 << 10, int(a.dim(0))));
          splitk_tests.insert(int(a.dim(0)) / std::min(1 << 11, int(a.dim(0))));
          splitk_tests.insert(int(a.dim(0)) / std::min(1 << 12, int(a.dim(0))));
      } else {
          splitk_tests = {1};
      }
      std::vector<int> splitk_tests_vec(splitk_tests.begin(), splitk_tests.end());
      std::sort(splitk_tests_vec.begin(), splitk_tests_vec.end(), [](auto a, auto b){return a > b;});
      for (auto spk : splitk_tests_vec){
          float total_time = 0.0;
          params.split_k_slices = spk;
          int actual_run = 0;
          for (int j = 0; j < num_run; ++j){
              auto ev_start = tv::CUDAEvent();
              auto ev_end = tv::CUDAEvent();
              ev_start.record(stream_int);
              GemmMain::matmul2(params);
              ev_end.record(stream_int);
              if (j > 0){
                  // skip first run
                  auto cur_time = tv::CUDAEvent::sync_and_duration(ev_start, ev_end);
                  total_time += cur_time;
                  actual_run++;
                  if (min_time > 0 && cur_time > min_time * 1.5){
                      // early skip for slow kernels
                      break;
                  }
              }
          }
          total_time /= actual_run;
          times.push_back(total_time);
          if (min_time < 0){
              min_time = total_time;
          }else{
              min_time = std::min(min_time, total_time);
          }
          all_profile_res.push_back(GemmTuneResult(desp, arch, spk));
      }
  }
  TV_ASSERT_RT_ERR(!all_profile_res.empty(), "can't find suitable algorithm");
  auto min_idx = std::min_element(times.begin(), times.end()) - times.begin();
  auto min_tune_res = all_profile_res[min_idx];
  {
      std::lock_guard<std::mutex> guard(mutex_);
      algo_cache_key_t key;
      if (hint & 4){
          key = std::make_tuple(int(a.dtype()), int(b.dtype()), int(c.dtype()), m, n);
          mn_cache_[key] = min_tune_res;
      }
      else if (hint & 2){
          key = std::make_tuple(int(a.dtype()), int(b.dtype()), int(c.dtype()), n, k);
          nk_dgrad_cache_[key] = min_tune_res;
      }
      else if (hint & 1){
          key = std::make_tuple(int(a.dtype()), int(b.dtype()), int(c.dtype()), n, k);
          nk_forward_cache_[key] = min_tune_res;
      }
      else{
          TV_THROW_RT_ERR("not implemented");
      }
  }
  return std::make_tuple(min_tune_res, times[min_idx]);
}
} // namespace gemmops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib