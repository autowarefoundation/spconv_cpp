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
void ConvGemmOps::implicit_gemm_backward(ExternalAllocator& allocator, ConvTuner& conv_tuner, tv::Tensor features, tv::Tensor filters, tv::Tensor out_bp, tv::Tensor pair_fwd, tv::Tensor pair_bwd, std::vector<tv::Tensor> pair_mask_fwd_splits, std::vector<tv::Tensor> pair_mask_bwd_splits, std::vector<tv::Tensor> mask_argsort_fwd_splits, std::vector<tv::Tensor> mask_argsort_bwd_splits, tv::Tensor mask_output_fwd, tv::Tensor masks, std::tuple<int, int> arch, int mask_width, bool is_subm, std::uintptr_t stream_int, tv::CUDAKernelTimer timer, bool auto_fp32_accum, bool fp32_accum, bool use_tf32)   {
  
  auto filters_shape = filters.shape();
  auto filters_shape_vec = filters.shape_vector();
  uint32_t* mask_ptr = masks.data_ptr<uint32_t>();
  int num_mask = masks.dim(0);
  int out_channel = filters.dim(0);
  int in_channel = filters.dim(-1);
  int num_split = pair_mask_fwd_splits.size();
  TV_ASSERT_RT_ERR(num_mask == num_split, "error");
  filters = filters.view(out_channel, -1, in_channel);
  int kv = filters.dim(1);
  tv::Tensor din;
  if (is_subm){
      din = allocator.empty("DIn", 
          features.shape_vector(), features.dtype(), features.device(), stream_int);
  }else{
      din = allocator.zeros("DIn", 
          features.shape_vector(), features.dtype(), features.device(), stream_int);
  }
  tv::Tensor dfilters = allocator.zeros("DFilters", 
      filters_shape_vec, filters.dtype(), filters.device(), stream_int);
  dfilters = dfilters.view(out_channel, -1, in_channel);
  constexpr auto kForwardInt = static_cast<int>(tv::gemm::ConvOpType::kForward);
  constexpr auto kBackwardInputInt = static_cast<int>(tv::gemm::ConvOpType::kBackwardInput);
  constexpr auto kBackwardWeightInt = static_cast<int>(tv::gemm::ConvOpType::kBackwardWeight);
  constexpr auto kChannelLastInt = static_cast<int>(tv::gemm::ConvLayoutType::kChannelLast);
  // auto arch = get_compute_capability();
  auto dgrad_tuned_res_exist = conv_tuner.get_tuned_algo(
      kBackwardInputInt,
      int(din.dtype()),
      int(filters.dtype()),
      int(out_bp.dtype()),
      out_channel, in_channel, arch);
  auto wgrad_tuned_res_exist = conv_tuner.get_tuned_algo(
      kBackwardWeightInt,
      int(features.dtype()),
      int(dfilters.dtype()),
      int(out_bp.dtype()),
      out_channel, in_channel, arch, mask_width);
  auto dgrad_tune_res = std::get<0>(dgrad_tuned_res_exist);
  auto dgrad_exists = std::get<1>(dgrad_tuned_res_exist);
  auto wgrad_tune_res = std::get<0>(wgrad_tuned_res_exist);
  auto wgrad_exists = std::get<1>(wgrad_tuned_res_exist);
  if (!dgrad_exists){
      tv::Tensor mask, mask_argsort;
      if (is_subm){
          mask = pair_mask_fwd_splits[0].type_view(tv::uint32);
          mask_argsort = mask_argsort_fwd_splits[0];
      }else{
          mask = pair_mask_bwd_splits[0].type_view(tv::uint32);
          mask_argsort = mask_argsort_bwd_splits[0];
      }
      auto tune_res_time = conv_tuner.tune_and_cache(
          kBackwardInputInt,
          din, filters, out_bp,
          kChannelLastInt,
          kChannelLastInt,
          kChannelLastInt,
          1, 1, 1, 
          arch,
          mask,
          mask_argsort,
          pair_bwd,
          is_subm, // reverse_mask
          mask_ptr[0], // mask_filter
          -1, // mask width
          tv::Tensor(), // mask_output
          1.0, 0.0,
          stream_int, 
          auto_fp32_accum,
          fp32_accum,
          5, // num_run
          use_tf32);
      dgrad_tune_res = std::get<0>(tune_res_time);
  }
  if (!wgrad_exists){
      auto tune_res_time = conv_tuner.tune_and_cache(
          kBackwardWeightInt,
          features, dfilters, out_bp,
          kChannelLastInt,
          kChannelLastInt,
          kChannelLastInt,
          1, 1, 1, 
          arch,
          mask_output_fwd[0].type_view(tv::uint32),
          mask_argsort_fwd_splits[0],
          pair_fwd,
          false, // reverse_mask
          mask_ptr[0], // mask_filter
          mask_width,
          tv::Tensor(), // mask_output
          1.0, 0.0,
          stream_int, 
          auto_fp32_accum,
          fp32_accum,
          5, // num_run
          use_tf32);
      wgrad_tune_res = std::get<0>(tune_res_time);
  }
  int ws_size = conv_tuner.query_workspace_size(wgrad_tune_res.algo_desp,
                                         wgrad_tune_res.splitk,
                                         kBackwardWeightInt,
                                         pair_fwd.dim(1), in_channel,
                                         out_channel, kv);
  ExternalAllocator::guard_t workspace_guard;
  tv::Tensor workspace;
  if (ws_size > 0){
      workspace_guard = allocator.empty_guard({int64_t(ws_size)}, tv::uint8, 0);
      workspace = workspace_guard->tensor;
  }
  for (int j = 0; j < num_split; ++j){
      tv::Tensor mask, mask_argsort;
      if (is_subm){
          mask = pair_mask_fwd_splits[j].type_view(tv::uint32);
          mask_argsort = mask_argsort_fwd_splits[j];
      }else{
          mask = pair_mask_bwd_splits[j].type_view(tv::uint32);
          mask_argsort = mask_argsort_bwd_splits[j];
      }
      float beta = j == 0 ? 0 : 1;
      conv_tuner.run_with_tuned_result(
          dgrad_tune_res,
          kBackwardInputInt,
          din,
          filters,
          out_bp,
          mask,
          mask_argsort,
          tv::Tensor(), // mask_output
          pair_bwd,
          is_subm, // reverse_mask
          mask_ptr[j],
          -1, // mask_width
          1.0, beta,
          stream_int,
          tv::Tensor(), // workspace
          false, // verbose
          timer);
      conv_tuner.run_with_tuned_result(
          wgrad_tune_res,
          kBackwardWeightInt,
          features, dfilters, out_bp,
          mask_output_fwd[j].type_view(tv::uint32),
          mask_argsort_fwd_splits[j],
          tv::Tensor(), // mask_output
          pair_fwd,
          false, // reverse_mask
          mask_ptr[j], // mask_filter
          mask_width,
          1.0, 0.0,
          stream_int, 
          workspace, // workspace
          false, // verbose
          timer);
  }
}
} // namespace spops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib