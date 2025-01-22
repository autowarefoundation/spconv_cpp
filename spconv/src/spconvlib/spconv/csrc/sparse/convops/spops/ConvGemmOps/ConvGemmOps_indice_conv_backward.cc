#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
#include <spconvlib/spconv/csrc/sparse/gather/GatherCPU.h>
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
using GatherCPU = spconvlib::spconv::csrc::sparse::gather::GatherCPU;
void ConvGemmOps::indice_conv_backward(ExternalAllocator& allocator, ExternalSpconvMatmul& ext_mm, GemmTuner& gemm_tuner, bool all_w_is_krsc, bool filter_hwio, tv::Tensor features, tv::Tensor filters, tv::Tensor out_bp, tv::Tensor indice_pairs, tv::Tensor indice_pair_num, std::tuple<int, int> arch, bool inverse, bool subm, int algo, std::uintptr_t stream_int, bool use_tf32)   {
  
  int kv_dim, out_channel, kv;
  std::vector<int64_t> filter_shape_per_kv;
  auto prev_filter_shape_vec = filters.shape_vector();
  bool is_KC_not_CK;
  if (!all_w_is_krsc){
      kv_dim = 0;
      is_KC_not_CK = !filter_hwio;
      if (filter_hwio){
          out_channel = filters.dim(-1);
          filter_shape_per_kv = {filters.dim(-2), out_channel};
      }else{
          out_channel = filters.dim(-2);
          filter_shape_per_kv = {out_channel, filters.dim(-1)};
      }
      filters = filters.view(-1, filters.dim(-2), filters.dim(-1));
      kv = filters.dim(0);
  }else{
      kv_dim = 1;
      out_channel = filters.dim(0);
      filters = filters.view(out_channel, -1, filters.dim(-1));
      is_KC_not_CK = true;
      kv = filters.dim(1);
      filter_shape_per_kv = {out_channel, filters.dim(-1)};
  }
  int kv_center = kv / 2;
  tv::Tensor din;
  auto dfilters = allocator.zeros("DFilters", 
          prev_filter_shape_vec, features.dtype(), features.device(), stream_int);
  dfilters = dfilters.view(filters.shape());
  if (subm){
      din = ext_mm.indice_conv_bwd_init_gemm("Features", 
          "Filters", "OutBp",
          "DFilters",
          all_w_is_krsc,
          is_KC_not_CK, kv_center);
  }else{
      din = allocator.zeros("DIn", 
              features.shape_vector(), features.dtype(), features.device(), stream_int);
  }
  if (kv == 1 && subm){
      return;
  }
  auto indice_pair_num_cpu = indice_pair_num.cpu();
  auto indice_pair_num_cpu_ptr = indice_pair_num_cpu.data_ptr<int>();
  int maxnhot = 0;
  bool all_zero = true;
  for (int i = 0; i < kv; ++i){
      if (indice_pair_num_cpu_ptr[i] != 0){
          indice_pair_num_cpu_ptr[i] = std::min(indice_pair_num_cpu_ptr[i], int(indice_pairs.dim(2)));
          all_zero = false;
          maxnhot = std::max(maxnhot, indice_pair_num_cpu_ptr[i]);
      }
  }
  if (subm && all_zero){
      return;
  }
  bool inited = subm;
  auto pair_in = indice_pairs[int(inverse)];
  auto pair_out = indice_pairs[int(!inverse)];
  if (features.is_cpu()){
      TV_ASSERT_RT_ERR(filters.is_cpu() && indice_pairs.is_cpu(), "error");
      auto inp_buffer = allocator.empty("InpBuffer", 
          {maxnhot, features.dim(1)}, features.dtype(), -1);
      auto out_buffer = allocator.empty("OutBuffer", 
          {maxnhot, out_bp.dim(1)}, out_bp.dtype(), -1);
      for (int i = 0; i < kv; ++i){
          int nhot = indice_pair_num_cpu_ptr[i];
          if (subm && i == kv_center){
              continue;
          }
          if (subm && i > kv_center){
              nhot = indice_pair_num_cpu_ptr[kv - i - 1];
          }
          if (nhot <= 0){
              continue;
          }
          auto inp_indices = pair_in[i].slice_first_axis(0, nhot);
          auto out_indices = pair_out[i].slice_first_axis(0, nhot);
          GatherCPU::gather(inp_buffer, features, inp_indices);
          GatherCPU::gather(out_buffer, out_bp, out_indices);
          ext_mm.indice_conv_bwd_cpu_gemm("InpBuffer", 
              "OutBuffer", 
              "Filters",
              "DFilters", all_w_is_krsc,
              is_KC_not_CK, nhot, i);
          GatherCPU::scatter_add(din, inp_buffer, inp_indices);
      }
      return;
  }
  int profile_idx = kv_center;
  if (subm)
      profile_idx = kv_center - 1;
  int nhot_profile = indice_pair_num_cpu_ptr[profile_idx];
  if (nhot_profile == 0){
      profile_idx = 0;
      for (int i = 0; i < kv; ++i){
          int nhot = indice_pair_num_cpu_ptr[i];
          if (nhot > nhot_profile){
              nhot_profile = nhot;
              profile_idx = i;
          }
      }
  }
  TV_ASSERT_RT_ERR(nhot_profile > 0, "this shouldn't happen");
  // auto arch = get_compute_capability();
  int sac_shuffle_type = static_cast<int>(tv::gemm::ShuffleStrideType::kShuffleAC);
  int sab_shuffle_type = static_cast<int>(tv::gemm::ShuffleStrideType::kShuffleAB);
  auto dgrad_tuned_res_exist = gemm_tuner.get_tuned_algo(
      int(out_bp.dtype()),
      int(filters.dtype()),
      int(din.dtype()),
      out_bp.shape_vector(),
      filter_shape_per_kv,
      din.shape_vector(),
      false,
      !is_KC_not_CK,
      false,
      arch,
      sac_shuffle_type,
      {nhot_profile},
      {},
      {nhot_profile},
      2);
  auto tuned_res_dgrad = std::get<0>(dgrad_tuned_res_exist);
  auto dgrad_exists = std::get<1>(dgrad_tuned_res_exist);
  if (!dgrad_exists){
      auto inp_indices = pair_in[profile_idx].slice_first_axis(0, nhot_profile);
      auto out_indices = pair_out[profile_idx].slice_first_axis(0, nhot_profile);
      auto filter = filters.select(kv_dim, profile_idx);
      auto tune_res_time = gemm_tuner.tune_and_cache(
          out_bp,
          filter,
          din,
          false,
          !is_KC_not_CK,
          false,
          arch,
          sac_shuffle_type,
          out_indices,
          tv::Tensor(),
          inp_indices,
          2,
          1.0,
          0.0,
          stream_int,
          5, // num_run
          use_tf32);
      tuned_res_dgrad = std::get<0>(tune_res_time);
  }
  tv::Tensor a_wgrad, b_wgrad;
  if (is_KC_not_CK){
      a_wgrad = out_bp;
      b_wgrad = features;
  }
  else{
      a_wgrad = features;
      b_wgrad = out_bp;
  }
  auto wgrad_tuned_res_exist = gemm_tuner.get_tuned_algo(
      int(a_wgrad.dtype()),
      int(b_wgrad.dtype()),
      int(filters.dtype()),
      a_wgrad.shape_vector(),
      b_wgrad.shape_vector(),
      filter_shape_per_kv,
      true,
      false,
      false,
      arch,
      sab_shuffle_type,
      {nhot_profile},
      {nhot_profile},
      {},
      4);
  auto tuned_res_wgrad = std::get<0>(wgrad_tuned_res_exist);
  auto wgrad_exists = std::get<1>(wgrad_tuned_res_exist);
  if (!wgrad_exists){
      auto inp_indices = pair_in[profile_idx].slice_first_axis(0, nhot_profile);
      auto out_indices = pair_out[profile_idx].slice_first_axis(0, nhot_profile);
      auto dfilter = dfilters.select(kv_dim, profile_idx);
      tv::Tensor a_inds_wgrad, b_inds_wgrad;
      if (is_KC_not_CK){
          a_inds_wgrad = out_indices;
          b_inds_wgrad = inp_indices;
      }else{
          a_inds_wgrad = inp_indices;
          b_inds_wgrad = out_indices;
      }
      auto tune_res_time = gemm_tuner.tune_and_cache(
          a_wgrad,
          b_wgrad,
          dfilter,
          true,
          false,
          false,
          arch,
          sab_shuffle_type,
          a_inds_wgrad,
          b_inds_wgrad,
          tv::Tensor(),
          4,
          1.0,
          0.0,
          stream_int,
          5, // num_run
          use_tf32);
      tuned_res_wgrad = std::get<0>(tune_res_time);
  }
  std::vector<int64_t> a_shape{maxnhot, out_bp.dim(1)};
  std::vector<int64_t> b_shape{maxnhot, features.dim(1)};
  if (!is_KC_not_CK){
      std::swap(a_shape, b_shape);
  }
  auto mnk = GemmTuner::extract_mnk_vector(a_shape, b_shape, 
      tuned_res_wgrad.algo_desp.trans_a(),
      tuned_res_wgrad.algo_desp.trans_b(),
      tuned_res_wgrad.algo_desp.trans_c(),
      sab_shuffle_type, 
      {maxnhot}, {maxnhot}, {});
  auto ws_size = tuned_res_wgrad.algo_desp.query_workspace_size(
      std::get<0>(mnk), std::get<1>(mnk), std::get<2>(mnk), tuned_res_wgrad.splitk);
  ExternalAllocator::guard_t workspace_guard;
  tv::Tensor workspace;
  if (ws_size > 0){
      workspace_guard = allocator.empty_guard({int64_t(ws_size)}, tv::uint8, 0);
      workspace = workspace_guard->tensor;
  }
  for (int i = 0; i < kv; ++i){
      int nhot = indice_pair_num_cpu_ptr[i];
      if (subm && i == kv_center){
          continue;
      }
      if (subm && i > kv_center){
          nhot = indice_pair_num_cpu_ptr[kv - i - 1];
      }
      if (nhot <= 0){
          continue;
      }
      auto inp_indices = pair_in[i].slice_first_axis(0, nhot);
      auto out_indices = pair_out[i].slice_first_axis(0, nhot);
      auto filter_i = filters.select(kv_dim, i);
      float beta = inited ? 1.0 : 0.0;
      gemm_tuner.run_with_tuned_result(
          tuned_res_dgrad,
          out_bp,
          filter_i,
          din,
          false,
          !is_KC_not_CK,
          false,
          arch,
          stream_int,
          sac_shuffle_type,
          out_indices,
          tv::Tensor(),
          inp_indices,
          2,
          1.0,
          beta);
      tv::Tensor a = out_bp;
      tv::Tensor b = features;
      tv::Tensor a_inds = out_indices;
      tv::Tensor b_inds = inp_indices;
      if (!is_KC_not_CK){
          std::swap(a, b);
          std::swap(a_inds, b_inds);
      }
      gemm_tuner.run_with_tuned_result(
          tuned_res_wgrad,
          a,
          b,
          dfilters.select(kv_dim, i),
          true,
          false,
          false,
          arch,
          stream_int,
          sab_shuffle_type,
          a_inds,
          b_inds,
          tv::Tensor(),
          4,
          1.0,
          beta);
      inited = true;
  }
}
} // namespace spops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib