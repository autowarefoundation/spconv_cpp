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
std::tuple<ConvTuneResult, float> ConvTunerSimple::tune_and_cache(int op_type, tv::Tensor inp, tv::Tensor weight, tv::Tensor output, int layout_i, int layout_w, int layout_o, int interleave_i, int interleave_w, int interleave_o, std::tuple<int, int> arch, tv::Tensor mask, tv::Tensor mask_argsort, tv::Tensor indices, bool reverse_mask, uint32_t mask_filter, int mask_width, tv::Tensor mask_output, float alpha, float beta, std::uintptr_t stream_int, bool auto_fp32_accum, bool fp32_accum, int num_run, bool use_tf32, tv::Tensor bias, tv::Tensor scale)   {
  
  TV_ASSERT_RT_ERR(num_run > 1, "error");
  auto avail = get_all_available(inp, weight, output, layout_i, layout_w,
                                 layout_o, interleave_i, interleave_w, interleave_o,
                                 arch, op_type, mask_width,
                                 auto_fp32_accum, fp32_accum, use_tf32,
                                 bias, scale);
  inp = inp.clone();
  weight = weight.clone();
  bool need_dynamic_mask = weight.dim(1) > 32;
  output = output.clone();
  int channel_k = output.dim(1);
  int channel_c = inp.dim(1);
  weight = weight.view(channel_k, -1, channel_c);
  std::vector<ConvTuneResult> all_profile_res;
  std::unordered_set<int> splitk_tests;
  std::vector<float> times;
  tv::gemm::ConvOpType op_type_cpp = static_cast<tv::gemm::ConvOpType>(op_type);
  float min_time = -1;
  for (auto& desp : avail){
      tv::gemm::ConvParams params(3, op_type_cpp, tv::CUDAKernelTimer(false));
      if (desp.is_nvrtc || prebuilt_names_.find(desp.__repr__()) == prebuilt_names_.end()){
          params.nvrtc_params = cached_get_nvrtc_params(desp, arch, stream_int);
      }
      params.conv_algo_desp = desp;
      params.input = inp;
      params.weight = weight.view(channel_k, -1, channel_c);
      params.output = output;
      params.mask_width = mask_width;
      params.alpha = alpha;
      params.beta = beta;
      params.stream = stream_int;
      params.mask_argsort = mask_argsort;
      params.indices = indices;
      params.mask = mask;
      params.mask_output = mask_output;
      if (desp.is_int8_inference){
          params.bias = bias;
          params.scale = scale;
      }
      // if (op_type_cpp == tv::gemm::ConvOpType::kBackwardWeight){
      //     TV_ASSERT_RT_ERR(!mask_output.empty(), "error");
      // }
      if (op_type_cpp == tv::gemm::ConvOpType::kBackwardInput){
          params.reverse_mask = reverse_mask;
      }
      params.mask_filter = mask_filter;
      if (desp.split_k_serial() && (op_type_cpp == tv::gemm::ConvOpType::kBackwardWeight)){
          splitk_tests = {1, 2, 4, 8, 16, 32, 64};
          splitk_tests.insert(int(inp.dim(0)) / std::min(1 << 10, int(inp.dim(0))));
          splitk_tests.insert(int(inp.dim(0)) / std::min(1 << 11, int(inp.dim(0))));
          splitk_tests.insert(int(inp.dim(0)) / std::min(1 << 12, int(inp.dim(0))));
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
              ConvMain::implicit_gemm2(params);
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
          all_profile_res.push_back(ConvTuneResult(desp, arch, spk));
      }
  }
  TV_ASSERT_RT_ERR(!all_profile_res.empty(), "can't find suitable algorithm for", op_type);
  auto min_idx = std::min_element(times.begin(), times.end()) - times.begin();
  auto min_tune_res = all_profile_res[min_idx];
  if (op_type_cpp != tv::gemm::ConvOpType::kBackwardWeight){
      mask_width = -1;
  }
  algo_cache_key_t key;
  key = std::make_tuple(int(inp.dtype()), int(weight.dtype()), 
      int(output.dtype()), channel_k, channel_c, std::get<0>(arch), std::get<1>(arch), mask_width, need_dynamic_mask);
  {
      std::lock_guard<std::mutex> guard(mutex_);
      if (op_type_cpp == tv::gemm::ConvOpType::kForward){
          kc_forward_cache_[key] = min_tune_res;
      }
      else if (op_type_cpp == tv::gemm::ConvOpType::kBackwardInput){
          kc_dgrad_cache_[key] = min_tune_res;
      }
      else if (op_type_cpp == tv::gemm::ConvOpType::kBackwardWeight){
          kc_wgrad_cache_[key] = min_tune_res;
      }
      else{
          TV_THROW_RT_ERR("not implemented");
      }
  }
  return std::make_tuple(min_tune_res, times[min_idx]);
}
} // namespace convops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib