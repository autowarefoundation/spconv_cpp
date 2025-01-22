#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
namespace spops {
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using GemmTuneResult = spconvlib::spconv::csrc::sparse::convops::GemmTuneResult;
using ConvTuneResult = spconvlib::spconv::csrc::sparse::convops::ConvTuneResult;
using ExternalSpconvMatmul = spconvlib::spconv::csrc::sparse::convops::ExternalSpconvMatmul;
using InferenceOps = spconvlib::spconv::csrc::sparse::inference::InferenceOps;
using GemmTuner = spconvlib::spconv::csrc::sparse::convops::gemmops::GemmTunerSimple;
using ConvTuner = spconvlib::spconv::csrc::sparse::convops::convops::ConvTunerSimple;
std::tuple<int, ConvTuneResult> ConvGemmOps::implicit_gemm(ExternalAllocator& allocator, ConvTuner& conv_tuner, tv::Tensor features, tv::Tensor filters, tv::Tensor pair_fwd, std::vector<tv::Tensor> pair_mask_fwd_splits, std::vector<tv::Tensor> mask_argsort_fwd_splits, int num_activate_out, tv::Tensor masks, std::tuple<int, int> arch, bool is_train, bool is_subm, std::uintptr_t stream_int, tv::CUDAKernelTimer timer, bool auto_fp32_accum, bool fp32_accum, tv::Tensor bias, float act_alpha, float act_beta, tv::gemm::Activation act_type, bool use_tf32, float output_scale, tv::Tensor scale, tv::Tensor output_add, float output_add_scale, int output_dtype)   {
  
  if (!bias.empty() || act_type != tv::gemm::Activation::kNone){
      TV_ASSERT_RT_ERR(pair_mask_fwd_splits.size() == 1, "SplitGemm don't support fused bias/act for now.");
  }
  uint32_t* mask_ptr = masks.data_ptr<uint32_t>();
  int num_mask = masks.dim(0);
  int out_channel = filters.dim(0);
  int in_channel = filters.dim(-1);
  int num_split = pair_mask_fwd_splits.size();
  TV_ASSERT_RT_ERR(num_mask == num_split, "error");
  filters = filters.view(out_channel, -1, in_channel);
  int kv = filters.dim(1);
  int mask_int_count = tv::div_up(kv, 32);
  tv::Tensor out_features;
  if (output_dtype < 0){
      output_dtype = int(features.dtype());
  }
  if (is_subm){
      out_features = allocator.empty("OutFeatures", 
          {num_activate_out, out_channel}, tv::DType(output_dtype), features.device(), stream_int, false /*is_temp*/, output_scale);
  }else{
      out_features = allocator.zeros("OutFeatures", 
          {num_activate_out, out_channel}, tv::DType(output_dtype), features.device(), stream_int, false /*is_temp*/, output_scale);
  }
  // auto start_ev = tv::CUDAEvent();
  // start_ev.record(stream_int);
  // auto arch = get_compute_capability();
  constexpr auto kForwardInt = static_cast<int>(tv::gemm::ConvOpType::kForward);
  constexpr auto kChannelLastInt = static_cast<int>(tv::gemm::ConvLayoutType::kChannelLast);
  auto tuned_res_exist = conv_tuner.get_tuned_algo(
      kForwardInt,
      int(features.dtype()),
      int(filters.dtype()),
      int(out_features.dtype()),
      out_channel, in_channel, arch);
  auto tune_res = std::get<0>(tuned_res_exist);
  auto exists = std::get<1>(tuned_res_exist);
  if (!exists){
      auto tune_res_time = conv_tuner.tune_and_cache(
          kForwardInt,
          features, filters, out_features,
          kChannelLastInt,
          kChannelLastInt,
          kChannelLastInt,
          1, 1, 1, 
          arch,
          pair_mask_fwd_splits[0].type_view(tv::uint32),
          mask_argsort_fwd_splits[0],
          pair_fwd,
          false, // reverse_mask
          mask_ptr[0], // mask_filter
          -1,
          tv::Tensor(), // mask_output
          1.0, 0.0,
          stream_int, 
          auto_fp32_accum,
          fp32_accum,
          5, // num_run
          use_tf32,
          bias,
          scale);
      tune_res = std::get<0>(tune_res_time);
  }
  float alpha = 1.0;
  if (tune_res.algo_desp.is_int8_inference){
      alpha = output_scale;
  }
  int mask_width = tune_res.algo_desp.tile_shape[0];
  tv::Tensor mask_output_fwd;
  std::vector<tv::Tensor> mask_output_fwd_splits;
  if (is_train){
      mask_output_fwd = allocator.empty("MaskOutputFwd", 
          {num_split, tv::div_up(num_activate_out, mask_width), mask_int_count}, 
          tv::uint32, features.device(), stream_int);
      for (int i = 0; i < num_split; ++i){
          mask_output_fwd_splits.push_back(mask_output_fwd[i]);
      }
  }else{
      for (int i = 0; i < num_split; ++i){
          mask_output_fwd_splits.push_back(tv::Tensor());
      }
  }
  for (int j = 0; j < num_split; ++j){
      float beta = j == 0 ? 0 : 1;
      if (!bias.empty() && !tune_res.algo_desp.is_int8_inference){
          // use source as bias
          beta = 1;
      }
      if (!output_add.empty() && tune_res.algo_desp.is_int8_inference){
          // use source as bias
          beta = output_add_scale / output_scale;
      }
      if (j > 0){
          bias = tv::Tensor();
      }
      conv_tuner.run_with_tuned_result(
          tune_res,
          kForwardInt,
          features,
          filters,
          out_features,
          pair_mask_fwd_splits[j].type_view(tv::uint32),
          mask_argsort_fwd_splits[j],
          mask_output_fwd_splits[j],
          pair_fwd,
          false, // reverse_mask
          mask_ptr[j],
          -1, // mask_width
          alpha, beta,
          stream_int,
          tv::Tensor(), // workspace
          false, // verbose
          timer, 
          false,
          bias,
          act_alpha,
          act_beta,
          act_type,
          scale,
          output_add);
  }
  // auto end_ev = tv::CUDAEvent();
  // end_ev.record(stream_int);
  // tv::ssprint(tune_res.algo_desp.__repr__(), "WTF", exists, 
  //     features.shape(), filters.shape(), out_features.shape(), tv::CUDAEvent::sync_and_duration(start_ev, end_ev));
  return std::make_tuple(mask_width, tune_res);
}
} // namespace spops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib