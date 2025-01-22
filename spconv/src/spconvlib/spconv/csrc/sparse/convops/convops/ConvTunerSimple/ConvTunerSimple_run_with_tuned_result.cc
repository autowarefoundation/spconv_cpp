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
void ConvTunerSimple::run_with_tuned_result(ConvTuneResult profile_res, int op_type, tv::Tensor inp, tv::Tensor weight, tv::Tensor output, tv::Tensor mask, tv::Tensor mask_argsort, tv::Tensor mask_output, tv::Tensor indices, bool reverse_mask, uint32_t mask_filter, int mask_width, float alpha, float beta, std::uintptr_t stream_int, tv::Tensor workspace, bool verbose, tv::CUDAKernelTimer timer, bool force_nvrtc, tv::Tensor bias, float act_alpha, float act_beta, tv::gemm::Activation act_type, tv::Tensor scale, tv::Tensor output_add)   {
  
  auto desp = profile_res.algo_desp;
  int split_k_slices = 1;
  if (profile_res.splitk > 1){
      split_k_slices = profile_res.splitk;
  }
  int channel_k = output.dim(1);
  int channel_c = inp.dim(1);
  tv::gemm::ConvOpType op_type_cpp = static_cast<tv::gemm::ConvOpType>(op_type);
  auto arch = profile_res.arch;
  tv::gemm::ConvParams params(3, op_type_cpp, timer);
  bool desp_is_static = prebuilt_names_.find(desp.__repr__()) == prebuilt_names_.end();
  if (force_nvrtc || (desp.is_nvrtc || desp_is_static)){
      params.nvrtc_params = cached_get_nvrtc_params(desp, arch, stream_int);
  }
  params.conv_algo_desp = desp;
  params.input = inp;
  params.weight = weight.view(channel_k, -1, channel_c);
  params.output = output;
  params.verbose = verbose;
  params.bias = bias;
  params.scale = scale;
  params.split_k_slices = split_k_slices;
  params.alpha = alpha;
  params.beta = beta;
  params.act_alpha = act_alpha;
  params.act_beta = act_beta;
  params.act_type = act_type;
  if (!output_add.empty() && desp.is_int8_inference){
      params.output_add = output_add;
  }
  params.stream = stream_int;
  params.mask_argsort = mask_argsort;
  params.indices = indices;
  params.mask = mask;
  params.mask_filter = mask_filter;
  params.mask_width = mask_width;
  params.mask_output = mask_output;
  params.reverse_mask = reverse_mask;
  if (timer.enable()){
      params.timer = timer;
  }
  params.workspace = workspace;
  ConvMain::implicit_gemm2(params);
}
} // namespace convops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib