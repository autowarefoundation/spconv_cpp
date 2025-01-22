#pragma once
#include <tensorview/profile/cuda_profiler.h>
#include <tensorview/utility/tuplehash.h>
#include <mutex>
#include <spconvlib/spconv/csrc/sparse/alloc/ExternalAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/ConvTuneResult.h>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/cumm/common/GemmBasicHost.h>
#include <spconvlib/cumm/common/CompileInfo.h>
#include <spconvlib/cumm/conv/main/ConvMainUnitTest.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
namespace convops {
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using ConvTuneResult = spconvlib::spconv::csrc::sparse::convops::ConvTuneResult;
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using CompileInfo = spconvlib::cumm::common::CompileInfo;
using ConvMain = spconvlib::cumm::conv::main::ConvMainUnitTest;
struct ConvTunerSimple {
  using static_key_t = std::tuple<int, int, int, int, int, int, int, int, int, int>;
  using algo_cache_key_t = std::tuple<int, int, int, int, int, int, int, int, bool>;
  std::vector<tv::gemm::ConvAlgoDesp> desps_;
  std::unordered_map<static_key_t, std::vector<tv::gemm::ConvAlgoDesp>> static_key_to_desps_;
  std::unordered_set<std::string> prebuilt_names_;
  std::mutex mutex_;
  std::unordered_map<algo_cache_key_t, ConvTuneResult> kc_forward_cache_;
  std::unordered_map<algo_cache_key_t, ConvTuneResult> kc_dgrad_cache_;
  std::unordered_map<algo_cache_key_t, ConvTuneResult> kc_wgrad_cache_;
  /**
   * @param desps 
   */
   ConvTunerSimple(std::vector<tv::gemm::ConvAlgoDesp> desps);
  /**
   * @param arch 
   */
  static std::vector<std::string> get_available_algo_str_from_arch(std::tuple<int, int> arch);
  /**
   * @param inp 
   * @param weight 
   * @param out 
   * @param layout_i 
   * @param layout_w 
   * @param layout_o 
   * @param interleave_i 
   * @param interleave_w 
   * @param interleave_o 
   * @param arch 
   * @param op_type 
   * @param mask_width 
   * @param auto_fp32_accum 
   * @param fp32_accum 
   * @param use_tf32 
   * @param bias 
   * @param scale 
   */
  std::vector<tv::gemm::ConvAlgoDesp> get_all_available(tv::Tensor inp, tv::Tensor weight, tv::Tensor out, int layout_i, int layout_w, int layout_o, int interleave_i, int interleave_w, int interleave_o, std::tuple<int, int> arch, int op_type, int mask_width, bool auto_fp32_accum, bool fp32_accum, bool use_tf32 = true, tv::Tensor bias = tv::Tensor(), tv::Tensor scale = tv::Tensor());
  /**
   * @param desp 
   * @param arch 
   * @param stream_int 
   */
  virtual tv::gemm::NVRTCParams cached_get_nvrtc_params(tv::gemm::ConvAlgoDesp desp, std::tuple<int, int> arch, std::uintptr_t stream_int);
  /**
   * @param op_type 
   * @param inp 
   * @param weight 
   * @param output 
   * @param layout_i 
   * @param layout_w 
   * @param layout_o 
   * @param interleave_i 
   * @param interleave_w 
   * @param interleave_o 
   * @param arch 
   * @param mask 
   * @param mask_argsort 
   * @param indices 
   * @param reverse_mask 
   * @param mask_filter 
   * @param mask_width 
   * @param mask_output 
   * @param alpha 
   * @param beta 
   * @param stream_int 
   * @param auto_fp32_accum 
   * @param fp32_accum 
   * @param num_run 
   * @param use_tf32 
   * @param bias 
   * @param scale 
   */
  std::tuple<ConvTuneResult, float> tune_and_cache(int op_type, tv::Tensor inp, tv::Tensor weight, tv::Tensor output, int layout_i, int layout_w, int layout_o, int interleave_i, int interleave_w, int interleave_o, std::tuple<int, int> arch, tv::Tensor mask, tv::Tensor mask_argsort, tv::Tensor indices, bool reverse_mask, uint32_t mask_filter = 0xffffffff, int mask_width = -1, tv::Tensor mask_output = tv::Tensor(), float alpha = 1.0, float beta = 0.0, std::uintptr_t stream_int = 0, bool auto_fp32_accum = true, bool fp32_accum = false, int num_run = 5, bool use_tf32 = true, tv::Tensor bias = tv::Tensor(), tv::Tensor scale = tv::Tensor());
  /**
   * @param op_type 
   * @param i_dtype 
   * @param w_dtype 
   * @param o_dtype 
   * @param k 
   * @param c 
   * @param arch 
   * @param mask_width 
   * @param need_dynamic_mask 
   */
  std::tuple<ConvTuneResult, bool> get_tuned_algo(int op_type, int i_dtype, int w_dtype, int o_dtype, int k, int c, std::tuple<int, int> arch, int mask_width = -1, bool need_dynamic_mask = false);
  /**
   * @param profile_res 
   * @param op_type 
   * @param inp 
   * @param weight 
   * @param output 
   * @param mask 
   * @param mask_argsort 
   * @param mask_output 
   * @param indices 
   * @param reverse_mask 
   * @param mask_filter 
   * @param mask_width 
   * @param alpha 
   * @param beta 
   * @param stream_int 
   * @param workspace 
   * @param verbose 
   * @param timer 
   * @param force_nvrtc 
   * @param bias 
   * @param act_alpha 
   * @param act_beta 
   * @param act_type 
   * @param scale 
   * @param output_add 
   */
  void run_with_tuned_result(ConvTuneResult profile_res, int op_type, tv::Tensor inp, tv::Tensor weight, tv::Tensor output, tv::Tensor mask, tv::Tensor mask_argsort, tv::Tensor mask_output, tv::Tensor indices, bool reverse_mask, uint32_t mask_filter = 0xffffffff, int mask_width = -1, float alpha = 1.0, float beta = 0.0, std::uintptr_t stream_int = 0, tv::Tensor workspace = tv::Tensor(), bool verbose = false, tv::CUDAKernelTimer timer = tv::CUDAKernelTimer(false), bool force_nvrtc = false, tv::Tensor bias = tv::Tensor(), float act_alpha = 0.0, float act_beta = 0.0, tv::gemm::Activation act_type = tv::gemm::Activation::kNone, tv::Tensor scale = tv::Tensor(), tv::Tensor output_add = tv::Tensor());
  /**
   * @param desp 
   * @param splitk 
   * @param op_type 
   * @param N 
   * @param C 
   * @param K 
   * @param kv 
   */
  int query_workspace_size(tv::gemm::ConvAlgoDesp desp, int splitk, int op_type, int N, int C, int K, int kv);
};
} // namespace convops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib