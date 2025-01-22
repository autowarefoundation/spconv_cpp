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
std::vector<tv::gemm::ConvAlgoDesp> ConvTunerSimple::get_all_available(tv::Tensor inp, tv::Tensor weight, tv::Tensor out, int layout_i, int layout_w, int layout_o, int interleave_i, int interleave_w, int interleave_o, std::tuple<int, int> arch, int op_type, int mask_width, bool auto_fp32_accum, bool fp32_accum, bool use_tf32, tv::Tensor bias, tv::Tensor scale)   {
  
  tv::gemm::ConvOpType op_type_cpp = static_cast<tv::gemm::ConvOpType>(op_type);
  bool is_fp16 = (inp.dtype() == tv::float16 && 
      weight.dtype() == tv::float16 && out.dtype() == tv::float16);
  bool use_f32_as_accum = false;
  int kv = 1;
  for (int i = 0; i < weight.ndim() - 2; ++i){
      kv *= weight.dim(i + 1);
  }
  if (is_fp16){
      if (auto_fp32_accum){
          if (op_type_cpp == tv::gemm::ConvOpType::kForward)
              use_f32_as_accum = weight.dim(-1) * kv > 128 * 27;
          else if (op_type_cpp == tv::gemm::ConvOpType::kBackwardInput)
              use_f32_as_accum = weight.dim(0) * kv > 128 * 27;
      }else{
          use_f32_as_accum = fp32_accum;
      }
  }
  use_f32_as_accum = false;
  std::vector<tv::gemm::ConvAlgoDesp> finally_algos;
  static_key_t static_key = std::make_tuple(
      layout_i, layout_w, layout_o,
      interleave_i, interleave_w, interleave_o, inp.dtype(),
      weight.dtype(), out.dtype(), op_type);
  if (static_key_to_desps_.find(static_key) == static_key_to_desps_.end()){
      return finally_algos;
  }
  auto& desps = static_key_to_desps_.at(static_key);
  for (auto& desp : desps){
      if (arch < desp.min_arch){
          continue;
      }
      if (arch >= std::make_tuple(7, 5) && desp.algo == "Volta"){
          continue;
      }
      if (!use_tf32){
          if (desp.tensorop[0] > 0 && inp.dtype() == tv::float32 && weight.dtype() == tv::float32 && out.dtype() == tv::float32){
              // tf32 op
              continue;
          }
      }
      if (arch >= std::make_tuple(7, 0) && is_fp16){
          // skip simt fp16 kernels if we have tensor core
          if (desp.algo == "Simt"){
              continue;
          }
          if (use_f32_as_accum){
              if (desp.dacc == tv::float16){
                  continue;
              }
          }
      }
      int ldi = inp.dim(-1);
      int ldw = weight.dim(-1);
      int ldo = out.dim(-1);
      bool mask_width_valid = true;
      if (desp.op_type == tv::gemm::ConvOpType::kBackwardWeight){
          TV_ASSERT_RT_ERR(mask_width > 0, "eroro");
          mask_width_valid = mask_width % desp.tile_shape[2] == 0;
      }
      bool require_dynamic_mask = kv > 32;
      if (desp.supported_ldx_conv(ldi, ldw, ldo) && mask_width_valid){
          if (!bias.empty() && !scale.empty()){
              TV_ASSERT_RT_ERR(bias.dtype() == scale.dtype(), "bias/scale dtype must equal to compute dtype in gemm");
              if (desp.dcomp != bias.dtype()){
                  continue;
              }
              if (!desp.is_int8_inference){
                  continue;
              }
          }else{
              if (desp.is_int8_inference){
                  continue;
              }
          }
          auto desp2 = desp;
          if (desp.is_nvrtc){
              if (!CompileInfo::algo_can_be_nvrtc_compiled(desp.min_arch)){
                  continue;
              }
          }
          if (!CompileInfo::arch_is_compiled_gemm(arch)){
              if (!CompileInfo::gemm_algo_can_use_ptx(desp.min_arch, arch)){
                  if (CompileInfo::algo_can_be_nvrtc_compiled(desp.min_arch)){
                      desp2.is_nvrtc = true;
                  }else{
                      continue;
                  }
              }
          }
          if (require_dynamic_mask){
              if (!desp.dynamic_mask){
                  continue;
              }
          }else{
              if (desp.dynamic_mask){
                  continue;
              }
          }
          finally_algos.push_back(desp2);
      }
  }
  std::sort(finally_algos.begin(), finally_algos.end(), [](auto a, auto b){return a.min_arch > b.min_arch;});
  return finally_algos;
}
} // namespace convops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib