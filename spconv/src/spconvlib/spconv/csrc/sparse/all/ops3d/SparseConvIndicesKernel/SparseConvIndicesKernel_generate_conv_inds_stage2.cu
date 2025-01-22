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
int SparseConvIndicesKernel::generate_conv_inds_stage2(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor indice_pairs_uniq_before_sort, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int num_out_act, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed, std::uintptr_t stream_int, bool use_bound_algo)   {
  
  auto custream = reinterpret_cast<cudaStream_t>(stream_int);
  // use_bound_algo = true;
  // TODO stream
  // TODO handle num input == 0
  int kv = ksize.op<tv::arrayops::prod>();
  TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");
  TV_ASSERT_RT_ERR(hashdata_k.dtype() == indice_pairs_uniq.dtype(), "error");
  TV_ASSERT_RT_ERR(hashdata_v.dtype() == tv::int32, "error");
  auto ctx = tv::Context();
  ctx.set_cuda_stream(custream);
  // indice_pairs: [2, kv, num_act_in_bounded]
  // indice_pairs_uniq: [indice_pairs.size() / 2 + 1]
  // out_inds: [MaxSize, 4]
  // auto timer = tv::CudaContextTimer<>();
  int64_t uniq_size = indice_pairs.size() / 2 + 1;
  TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) >= num_out_act, "error");
  TV_ASSERT_RT_ERR(out_inds.dim(0) >= num_out_act && out_inds.dim(1) == 4, "error");
  tv::cuda::Launch launcher_num_act_in(indices.dim(0), custream);
  launcher_num_act_in.blocks.y = kv;
  ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
  bool use_int32 = problem.check_npq_not_overflow();
  // TODO handle invalid num_out_act
  indice_pairs_uniq = indice_pairs_uniq.slice_first_axis(0, num_out_act);
  tv::cuda::Launch lanucher_build_hash(num_out_act, custream);
  tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){
    using V = int32_t;
    using K = TV_DECLTYPE(I);
    using table_t =
        tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                    tv::hash::default_empty_key_v<K>, false>;
    TV_ASSERT_RT_ERR(hashdata_k.dim(0) >= num_out_act, "hash size not enough");
    table_t hash = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
    tv::hash::clear_map_split(hash, custream);
    // hash.clear(custream);
    if (int(use_int32) == 0){
      ConvLocIter64 loc_iter(problem);
      lanucher_build_hash(build_conv_hash_table<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, hash, 
          out_inds.data_ptr<int>(), indice_pairs_uniq.data_ptr<const K>(), 
          loc_iter.layout_npq, num_out_act);
    }
    else if (int(use_int32) == 1){
      ConvLocIter loc_iter(problem);
      lanucher_build_hash(build_conv_hash_table<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, hash, 
          out_inds.data_ptr<int>(), indice_pairs_uniq.data_ptr<const K>(), 
          loc_iter.layout_npq, num_out_act);
    }
    else{
      TV_THROW_RT_ERR("unknown val int(use_int32), available: [0, 1]")
    }
    if (!use_bound_algo){
        launcher_num_act_in(calc_conv_indices_stage2<table_t>, hash, 
            indice_pairs_uniq_before_sort.data_ptr<const K>(),
            indice_pairs[1].data_ptr<int>(), 
            indices.dim(0), 
            indice_pairs.dim(2));
    }else{
        indice_num_per_loc.zero_(ctx);
        // copy previous pair in to indice_pairs_uniq
        // we need to ensure size of indice_pairs_uniq larger than pair in
        TV_ASSERT_RT_ERR(true, "error");
        tv::Tensor indice_pairs_in_temp = tv::from_blob(indice_pairs_uniq.raw_data(), {indice_pairs.dim(1), indice_pairs.dim(2)}, 
            indice_pairs.dtype(), indice_pairs.device());
        indice_pairs_in_temp.copy_(indice_pairs[0].view(-1), ctx);
        launcher_num_act_in(calc_conv_indices_stage2_bounded<table_t>, hash, 
            indice_pairs_uniq_before_sort.data_ptr<const K>(),
            indice_pairs_in_temp.data_ptr<const int>(),
            indice_pairs[0].data_ptr<int>(), 
            indice_pairs[1].data_ptr<int>(), 
            indice_num_per_loc.data_ptr<int>(),
            indices.dim(0), 
            indice_pairs.dim(2));
    }
  });
  return num_out_act;
}
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib