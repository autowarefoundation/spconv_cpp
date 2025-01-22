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
void GemmTunerSimple::run_with_tuned_result(GemmTuneResult profile_res, tv::Tensor a, tv::Tensor b, tv::Tensor c, bool trans_a, bool trans_b, bool trans_c, std::tuple<int, int> arch, std::uintptr_t stream_int, int shuffle_type, tv::Tensor a_inds, tv::Tensor b_inds, tv::Tensor c_inds, int hint, float alpha, float beta, tv::Tensor workspace, tv::CUDAKernelTimer timer, bool force_nvrtc, tv::Tensor bias, float act_alpha, float act_beta, tv::gemm::Activation act_type)   {
  
  auto& desp = profile_res.algo_desp;
  int split_k_slices = 1;
  if (profile_res.splitk > 1){
      split_k_slices = profile_res.splitk;
  }
  tv::gemm::GemmParams params;
  bool desp_is_static = prebuilt_names_.find(desp.__repr__()) == prebuilt_names_.end();
  if (force_nvrtc || (desp.is_nvrtc || desp_is_static)){
      params.nvrtc_params = cached_get_nvrtc_params(desp, profile_res.arch, stream_int);
  }
  params.a = a;
  params.b = b;
  params.c = c;
  params.d = bias;
  params.a_inds = a_inds;
  params.b_inds = b_inds;
  params.c_inds = c_inds;
  params.algo_desp = desp;
  params.split_k_slices = split_k_slices;
  params.stream = stream_int;
  params.alpha = alpha;
  params.beta = beta;
  params.act_alpha = act_alpha;
  params.act_beta = act_beta;
  params.act_type = act_type;
  params.workspace = workspace;
  GemmMain::matmul2(params);
}
} // namespace gemmops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib