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
void ConvGemmOps::indice_conv(ExternalAllocator& allocator, ExternalSpconvMatmul& ext_mm, GemmTuner& gemm_tuner, bool all_w_is_krsc, bool filter_hwio, tv::Tensor features, tv::Tensor filters, tv::Tensor indice_pairs, tv::Tensor indice_pair_num, std::tuple<int, int> arch, int num_activate_out, bool inverse, bool subm, int algo, std::uintptr_t stream_int, tv::Tensor bias, float act_alpha, float act_beta, tv::gemm::Activation act_type, bool use_tf32)   {
  
  int kv_dim, out_channel, kv;
  std::vector<int64_t> filter_shape_per_kv;
  bool is_KC_not_CK;
  bool has_bias = !bias.empty();
  bool has_act = act_type != tv::gemm::Activation::kNone;
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
  tv::Tensor out_features;
  if (subm){
      out_features = ext_mm.indice_conv_init_gemm("Features", 
          "Filters", all_w_is_krsc,
          is_KC_not_CK, kv_center, out_channel);
  }else{
      out_features = allocator.zeros("OutFeatures", 
          {num_activate_out, out_channel}, features.dtype(), features.device(), stream_int);
  }
  if (has_act || has_bias){
      TV_ASSERT_RT_ERR(!features.is_cpu(), "bias and act don't support cpu.");
  }
  if (kv == 1 && subm){
      if (has_bias && has_act){
          InferenceOps::bias_add_act_inplace(out_features, bias, act_type, act_alpha, act_beta, stream_int);
      }else{
          if (has_bias){
              InferenceOps::bias_add_inplace(out_features, bias, stream_int);
          }
          if (has_act){
              InferenceOps::activation_inplace(out_features, act_type, act_alpha, act_beta, stream_int);
          }
      }
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
  auto a = features;
  auto c = out_features;
  auto pair_in = indice_pairs[int(inverse)];
  auto pair_out = indice_pairs[int(!inverse)];
  if (features.is_cpu()){
      TV_ASSERT_RT_ERR(filters.is_cpu() && indice_pairs.is_cpu(), "error");
      auto inp_buffer = allocator.empty("InpBuffer", 
          {maxnhot, features.dim(1)}, features.dtype(), -1);
      auto out_buffer = allocator.empty("OutBuffer", 
          {maxnhot, out_features.dim(1)}, out_features.dtype(), -1);
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
          GatherCPU::gather(inp_buffer, a, inp_indices);
          ext_mm.indice_conv_cpu_gemm("InpBuffer", 
              "OutBuffer",
              "Filters", all_w_is_krsc,
              is_KC_not_CK, nhot, i);
          GatherCPU::scatter_add(c, out_buffer, out_indices);
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
  auto a_shape = a.shape();
  auto c_shape = c.shape();
  int sac_shuffle_type = static_cast<int>(tv::gemm::ShuffleStrideType::kShuffleAC);
  auto tuned_res_exist = gemm_tuner.get_tuned_algo(
      int(a.dtype()),
      int(filters.dtype()),
      int(c.dtype()),
      std::vector<int64_t>(a_shape.begin(), a_shape.end()),
      filter_shape_per_kv,
      std::vector<int64_t>(c_shape.begin(), c_shape.end()),
      false,
      is_KC_not_CK,
      false,
      arch,
      sac_shuffle_type,
      {nhot_profile},
      {},
      {nhot_profile},
      1);
  auto tune_res = std::get<0>(tuned_res_exist);
  auto exists = std::get<1>(tuned_res_exist);
  if (!exists){
      auto inp_indices = pair_in[profile_idx].slice_first_axis(0, nhot_profile);
      auto out_indices = pair_out[profile_idx].slice_first_axis(0, nhot_profile);
      auto filter = filters.select(kv_dim, profile_idx);
      auto tune_res_time = gemm_tuner.tune_and_cache(
          a,
          filter,
          c,
          false,
          is_KC_not_CK,
          false,
          arch,
          sac_shuffle_type,
          inp_indices,
          tv::Tensor(),
          out_indices,
          1,
          1.0,
          0.0,
          stream_int,
          5, // num_run
          use_tf32);
      tune_res = std::get<0>(tune_res_time);
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
      auto b = filters.select(kv_dim, i);
      float beta = inited ? 1.0 : 0.0;
      gemm_tuner.run_with_tuned_result(
          tune_res,
          a,
          b,
          c,
          false,
          is_KC_not_CK,
          false,
          arch,
          stream_int,
          sac_shuffle_type,
          inp_indices,
          tv::Tensor(),
          out_indices,
          1,
          1.0,
          beta);
      inited = true;
  }
  if (has_bias && has_act){
      InferenceOps::bias_add_act_inplace(out_features, bias, act_type, act_alpha, act_beta, stream_int);
  }else{
      if (has_bias){
          InferenceOps::bias_add_inplace(out_features, bias, stream_int);
      }
      if (has_act){
          InferenceOps::activation_inplace(out_features, act_type, act_alpha, act_beta, stream_int);
      }
  }
}
} // namespace spops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib