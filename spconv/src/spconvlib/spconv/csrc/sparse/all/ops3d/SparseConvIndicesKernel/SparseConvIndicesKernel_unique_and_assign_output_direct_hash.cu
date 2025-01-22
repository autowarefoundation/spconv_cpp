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
int SparseConvIndicesKernel::unique_and_assign_output_direct_hash(tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor uniq_cnt, tv::Tensor out_inds, int num_out_bound, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, std::uintptr_t stream_int)   {
  
  auto custream = reinterpret_cast<cudaStream_t>(stream_int);
  tv::cuda::Launch lanucher_build_hash(hashdata_k.size(), custream);
  ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
  bool use_int32 = problem.check_npq_not_overflow();
  auto tvctx = tv::Context();
  tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream_int));
  if (num_out_bound <= 0){
      num_out_bound = hashdata_k.size();
  }
  if (int(use_int32) == 0){
    ConvLocIter64 loc_iter(problem);
    tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){
        using V = int32_t;
        using K = TV_DECLTYPE(I);
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
        lanucher_build_hash(arange_hash_table_and_assign_out<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, table, 
            out_inds.data_ptr<int>(), uniq_cnt.data_ptr<int>(), num_out_bound,
            loc_iter.layout_npq);
    });
  }
  else if (int(use_int32) == 1){
    ConvLocIter loc_iter(problem);
    tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){
        using V = int32_t;
        using K = TV_DECLTYPE(I);
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
        lanucher_build_hash(arange_hash_table_and_assign_out<table_t, std::decay_t<decltype(loc_iter.layout_npq)>>, table, 
            out_inds.data_ptr<int>(), uniq_cnt.data_ptr<int>(), num_out_bound,
            loc_iter.layout_npq);
    });
  }
  else{
    TV_THROW_RT_ERR("unknown val int(use_int32), available: [0, 1]")
  }
  auto uniq_cnt_cpu = uniq_cnt.cpu(tvctx);
  return std::min(uniq_cnt_cpu.data_ptr<int>()[0], num_out_bound);
}
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib