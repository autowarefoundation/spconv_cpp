#pragma once
#include <tensorview/profile/cuda_profiler.h>
#include <tensorview/utility/tuplehash.h>
#include <mutex>
#include <spconvlib/spconv/csrc/sparse/alloc/ExternalAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/GemmTuneResult.h>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/cumm/common/GemmBasicHost.h>
#include <spconvlib/cumm/common/CompileInfo.h>
#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
namespace gemmops {
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using GemmTuneResult = spconvlib::spconv::csrc::sparse::convops::GemmTuneResult;
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using CompileInfo = spconvlib::cumm::common::CompileInfo;
using GemmMain = spconvlib::cumm::gemm::main::GemmMainUnitTest;
struct GemmTunerSimple {
  using static_key_t = std::tuple<bool, bool, bool, int, int, int, int>;
  using algo_cache_key_t = std::tuple<int, int, int, int, int>;
  std::vector<tv::gemm::GemmAlgoDesp> desps_;
  std::unordered_map<static_key_t, std::vector<tv::gemm::GemmAlgoDesp>> static_key_to_desps_;
  std::unordered_set<std::string> prebuilt_names_;
  std::mutex mutex_;
  std::unordered_map<algo_cache_key_t, GemmTuneResult> nk_forward_cache_;
  std::unordered_map<algo_cache_key_t, GemmTuneResult> nk_dgrad_cache_;
  std::unordered_map<algo_cache_key_t, GemmTuneResult> mn_cache_;
  /**
   * @param desps 
   */
   GemmTunerSimple(std::vector<tv::gemm::GemmAlgoDesp> desps);
  /**
   * @param arch 
   */
  static std::vector<std::string> get_available_algo_str_from_arch(std::tuple<int, int> arch);
  /**
   * @param a 
   * @param b 
   * @param c 
   * @param trans_a 
   * @param trans_b 
   * @param trans_c 
   * @param arch 
   * @param shuffle_type 
   * @param use_tf32 
   */
  std::vector<tv::gemm::GemmAlgoDesp> get_all_available(tv::Tensor a, tv::Tensor b, tv::Tensor c, bool trans_a, bool trans_b, bool trans_c, std::tuple<int, int> arch, int shuffle_type, bool use_tf32 = true);
  /**
   * @param a_shape 
   * @param b_shape 
   * @param trans_a 
   * @param trans_b 
   * @param trans_c 
   * @param arch 
   * @param shuffle_type 
   * @param a_inds_shape 
   * @param b_inds_shape 
   * @param c_inds_shape 
   * @param hint 
   */
  std::tuple<int, int, int> extract_mnk(tv::TensorShape a_shape, tv::TensorShape b_shape, bool trans_a, bool trans_b, bool trans_c, std::tuple<int, int> arch, int shuffle_type, tv::TensorShape a_inds_shape, tv::TensorShape b_inds_shape, tv::TensorShape c_inds_shape, int hint = 0);
  /**
   * @param a_shape 
   * @param b_shape 
   * @param trans_a 
   * @param trans_b 
   * @param trans_c 
   * @param shuffle_type 
   * @param a_inds_shape 
   * @param b_inds_shape 
   * @param c_inds_shape 
   */
  static std::tuple<int, int, int> extract_mnk_vector(std::vector<int64_t> a_shape, std::vector<int64_t> b_shape, bool trans_a, bool trans_b, bool trans_c, int shuffle_type, std::vector<int64_t> a_inds_shape, std::vector<int64_t> b_inds_shape, std::vector<int64_t> c_inds_shape);
  /**
   * @param desp 
   * @param arch 
   * @param stream_int 
   */
  virtual tv::gemm::NVRTCParams cached_get_nvrtc_params(tv::gemm::GemmAlgoDesp desp, std::tuple<int, int> arch, std::uintptr_t stream_int);
  /**
   * @param a 
   * @param b 
   * @param c 
   * @param trans_a 
   * @param trans_b 
   * @param trans_c 
   * @param arch 
   * @param shuffle_type 
   * @param a_inds 
   * @param b_inds 
   * @param c_inds 
   * @param hint 
   * @param alpha 
   * @param beta 
   * @param stream_int 
   * @param num_run 
   * @param use_tf32 
   */
  std::tuple<GemmTuneResult, float> tune_and_cache(tv::Tensor a, tv::Tensor b, tv::Tensor c, bool trans_a, bool trans_b, bool trans_c, std::tuple<int, int> arch, int shuffle_type, tv::Tensor a_inds, tv::Tensor b_inds, tv::Tensor c_inds, int hint = 0, float alpha = 1.0, float beta = 0.0, std::uintptr_t stream_int = 0, int num_run = 5, bool use_tf32 = true);
  /**
   * @param a_dtype 
   * @param b_dtype 
   * @param c_dtype 
   * @param a_shape 
   * @param b_shape 
   * @param c_shape 
   * @param trans_a 
   * @param trans_b 
   * @param trans_c 
   * @param arch 
   * @param shuffle_type 
   * @param a_inds_shape 
   * @param b_inds_shape 
   * @param c_inds_shape 
   * @param hint 
   */
  std::tuple<GemmTuneResult, bool> get_tuned_algo(int a_dtype, int b_dtype, int c_dtype, std::vector<int64_t> a_shape, std::vector<int64_t> b_shape, std::vector<int64_t> c_shape, bool trans_a, bool trans_b, bool trans_c, std::tuple<int, int> arch, int shuffle_type, std::vector<int64_t> a_inds_shape, std::vector<int64_t> b_inds_shape, std::vector<int64_t> c_inds_shape, int hint = 0);
  /**
   * @param profile_res 
   * @param a 
   * @param b 
   * @param c 
   * @param trans_a 
   * @param trans_b 
   * @param trans_c 
   * @param arch 
   * @param stream_int 
   * @param shuffle_type 
   * @param a_inds 
   * @param b_inds 
   * @param c_inds 
   * @param hint 
   * @param alpha 
   * @param beta 
   * @param workspace 
   * @param timer 
   * @param force_nvrtc 
   * @param bias 
   * @param act_alpha 
   * @param act_beta 
   * @param act_type 
   */
  void run_with_tuned_result(GemmTuneResult profile_res, tv::Tensor a, tv::Tensor b, tv::Tensor c, bool trans_a, bool trans_b, bool trans_c, std::tuple<int, int> arch, std::uintptr_t stream_int, int shuffle_type, tv::Tensor a_inds, tv::Tensor b_inds, tv::Tensor c_inds, int hint = 0, float alpha = 1.0, float beta = 0.0, tv::Tensor workspace = tv::Tensor(), tv::CUDAKernelTimer timer = tv::CUDAKernelTimer(false), bool force_nvrtc = false, tv::Tensor bias = tv::Tensor(), float act_alpha = 0.0, float act_beta = 0.0, tv::gemm::Activation act_type = tv::gemm::Activation::kNone);
};
} // namespace gemmops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib