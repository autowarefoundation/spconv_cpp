#include <spconvlib/spconv/csrc/sparse/all/ops3d/SparseConvIndicesKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {
using TensorView = spconvlib::cumm::common::TensorView;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
using TensorViewHashKernel = spconvlib::cumm::common::TensorViewHashKernel;
using ConvLocIter = spconvlib::spconv::csrc::sparse::all::ops3d::spinds::ConvOutLocIter;
using ConvProblem = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::spinds::ConvProblem;
using ConvLocIter64 = spconvlib::spconv::csrc::sparse::all::ops3d::spinds64::ConvOutLocIter;
int SparseConvIndicesKernel::generate_subm_conv_inds(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> dilation, tv::Tensor indice_pair_mask, bool is_train, std::uintptr_t stream_int)   {
  
  int num_act_in_real = indices.dim(0);
  auto custream = reinterpret_cast<cudaStream_t>(stream_int);
  auto ctx = tv::Context();
  ctx.set_cuda_stream(custream);
  if (!indice_pair_mask.empty()){
      // TV_ASSERT_INVALID_ARG(ksize.op<tv::arrayops::prod>() <= 32, "for now only support 32bit mask");
  }
  // TODO stream
  // TODO handle num input == 0
  tv::array<int, 3> stride, padding;
  for (int i = 0; i < 3; ++i){
      TV_ASSERT_RT_ERR(ksize[i] % 2 == 1, "subm only support odd ksize");
      stride[i] = 1;
      padding[i] = (ksize[i] / 2) * dilation[i];
  }
  int kv = ksize.op<tv::arrayops::prod>();
  int mask_int_count = tv::div_up(kv, 32);
  TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");
  // indice_pairs: [1 or 2, kv, num_act_in] if mask else [2, kv, num_act_in]
  // out_inds: [MaxSize, 4]
  TV_ASSERT_RT_ERR(indice_num_per_loc.dim(0) == kv, "error");
  tv::cuda::Launch launcher_num_act_in(num_act_in_real, custream);
  launcher_num_act_in.blocks.y = (kv / 2) + 1;
  // launcher_num_act_in.blocks.y = kv;
  ConvProblem problem(batch_size, 1, 1, input_dims, input_dims, ksize, padding, stride, dilation);
  bool use_int32 = problem.check_npq_not_overflow();
  tv::cuda::Launch lanucher_build_hash(num_act_in_real, custream);
  if (int(use_int32) == 0){
        ConvLocIter64 loc_iter(problem);
        tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){
            using V = int32_t;
            using K = TV_DECLTYPE(I);
            TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<K>::max(), 
                "kernel volume must smaller than max value of K");
            using table_t =
                tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                            tv::hash::default_empty_key_v<K>, false>;
            TV_ASSERT_RT_ERR(hashdata_k.dim(0) >= num_act_in_real, "hash size not enough");
            table_t hash = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
            tv::hash::clear_map_split(hash, custream);
            lanucher_build_hash(build_subm_conv_hash_table<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, hash, indices.data_ptr<const int>(),
                loc_iter.layout_npq, num_act_in_real);
            if (!indice_pair_mask.empty()){
                TV_ASSERT_RT_ERR(indice_pairs.ndim() == 3, "error");
                TV_ASSERT_RT_ERR(indice_pairs.dim(0) == (is_train ? 2 : 1), "error");
                TV_ASSERT_INVALID_ARG(indice_pair_mask.ndim() == 3, "error");
                // indice_pair_mask: [mask_split_count, num_act_in, num_mask_per_point]
                if (indice_pair_mask.dim(0) == 2){
                    auto mask_0 = indice_pair_mask[0].slice_first_axis(0, num_act_in_real);
                    auto mask_1 = indice_pair_mask[1].slice_first_axis(0, num_act_in_real);
                    tv::cuda::Launch lanucher_fill(num_act_in_real, custream);
                    lanucher_fill(cudakers::fill_kernel<uint32_t>, mask_0.data_ptr<uint32_t>(), (1 << (kv / 2)), indices.dim(0));
                    mask_1.zero_(ctx);
                    auto kernel = &calc_subm_conv_indices_split_mask<table_t, ConvLocIter64>;
                    launcher_num_act_in(kernel, loc_iter, hash,  
                        indices.data_ptr<const int>(), indice_pairs.data_ptr<int>(), 
                        mask_0.data_ptr<uint32_t>(), mask_1.data_ptr<uint32_t>(), 
                        indices.dim(0), indice_pairs.dim(2), kv, is_train);
                }else{
                    // indice_pair_mask: [1, num_act_in, num_mask_per_point]
                    tv::cuda::Launch lanucher_fill(num_act_in_real, custream);
                    if (mask_int_count == 1){
                        lanucher_fill(cudakers::fill_kernel<uint32_t>, indice_pair_mask.data_ptr<uint32_t>(), (1 << (kv / 2)), indices.dim(0));
                    }
                    else{
                        lanucher_fill(init_subm_multiple_mask_int_kernel<uint32_t>, 
                                indice_pair_mask.data_ptr<uint32_t>(), kv / 2, indices.dim(0), mask_int_count);
                    }
                    TV_ASSERT_RT_ERR(indice_pair_mask.dim(0) == 1, "error");
                    launcher_num_act_in(calc_subm_conv_indices_mask<table_t, ConvLocIter64>, loc_iter, hash, 
                        indices.data_ptr<const int>(), indice_pairs.data_ptr<int>(), 
                        indice_pair_mask.data_ptr<uint32_t>(), indices.dim(0), indice_pairs.dim(2), kv, is_train, mask_int_count);
                }
            }else{
                TV_ASSERT_RT_ERR(indice_pairs.ndim() == 3, "error");
                TV_ASSERT_RT_ERR(indice_pairs.dim(0) == 2, "error");
                launcher_num_act_in(calc_subm_conv_indices<table_t, ConvLocIter64>, loc_iter, hash, indices.data_ptr<const int>(), 
                    indice_pairs.data_ptr<int>(), 
                    indice_num_per_loc.data_ptr<int>(), indices.dim(0), indice_pairs.dim(2), kv);
            }
        });
    return indices.dim(0);
  }
  else if (int(use_int32) == 1){
        ConvLocIter loc_iter(problem);
        tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){
            using V = int32_t;
            using K = TV_DECLTYPE(I);
            TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<K>::max(), 
                "kernel volume must smaller than max value of K");
            using table_t =
                tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                            tv::hash::default_empty_key_v<K>, false>;
            TV_ASSERT_RT_ERR(hashdata_k.dim(0) >= num_act_in_real, "hash size not enough");
            table_t hash = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
            tv::hash::clear_map_split(hash, custream);
            lanucher_build_hash(build_subm_conv_hash_table<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, hash, indices.data_ptr<const int>(),
                loc_iter.layout_npq, num_act_in_real);
            if (!indice_pair_mask.empty()){
                TV_ASSERT_RT_ERR(indice_pairs.ndim() == 3, "error");
                TV_ASSERT_RT_ERR(indice_pairs.dim(0) == (is_train ? 2 : 1), "error");
                TV_ASSERT_INVALID_ARG(indice_pair_mask.ndim() == 3, "error");
                // indice_pair_mask: [mask_split_count, num_act_in, num_mask_per_point]
                if (indice_pair_mask.dim(0) == 2){
                    auto mask_0 = indice_pair_mask[0].slice_first_axis(0, num_act_in_real);
                    auto mask_1 = indice_pair_mask[1].slice_first_axis(0, num_act_in_real);
                    tv::cuda::Launch lanucher_fill(num_act_in_real, custream);
                    lanucher_fill(cudakers::fill_kernel<uint32_t>, mask_0.data_ptr<uint32_t>(), (1 << (kv / 2)), indices.dim(0));
                    mask_1.zero_(ctx);
                    auto kernel = &calc_subm_conv_indices_split_mask<table_t, ConvLocIter>;
                    launcher_num_act_in(kernel, loc_iter, hash,  
                        indices.data_ptr<const int>(), indice_pairs.data_ptr<int>(), 
                        mask_0.data_ptr<uint32_t>(), mask_1.data_ptr<uint32_t>(), 
                        indices.dim(0), indice_pairs.dim(2), kv, is_train);
                }else{
                    // indice_pair_mask: [1, num_act_in, num_mask_per_point]
                    tv::cuda::Launch lanucher_fill(num_act_in_real, custream);
                    if (mask_int_count == 1){
                        lanucher_fill(cudakers::fill_kernel<uint32_t>, indice_pair_mask.data_ptr<uint32_t>(), (1 << (kv / 2)), indices.dim(0));
                    }
                    else{
                        lanucher_fill(init_subm_multiple_mask_int_kernel<uint32_t>, 
                                indice_pair_mask.data_ptr<uint32_t>(), kv / 2, indices.dim(0), mask_int_count);
                    }
                    TV_ASSERT_RT_ERR(indice_pair_mask.dim(0) == 1, "error");
                    launcher_num_act_in(calc_subm_conv_indices_mask<table_t, ConvLocIter>, loc_iter, hash, 
                        indices.data_ptr<const int>(), indice_pairs.data_ptr<int>(), 
                        indice_pair_mask.data_ptr<uint32_t>(), indices.dim(0), indice_pairs.dim(2), kv, is_train, mask_int_count);
                }
            }else{
                TV_ASSERT_RT_ERR(indice_pairs.ndim() == 3, "error");
                TV_ASSERT_RT_ERR(indice_pairs.dim(0) == 2, "error");
                launcher_num_act_in(calc_subm_conv_indices<table_t, ConvLocIter>, loc_iter, hash, indices.data_ptr<const int>(), 
                    indice_pairs.data_ptr<int>(), 
                    indice_num_per_loc.data_ptr<int>(), indices.dim(0), indice_pairs.dim(2), kv);
            }
        });
    return indices.dim(0);
  }
  else{
    TV_THROW_RT_ERR("unknown val int(use_int32), available: [0, 1]")
  }
}
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib