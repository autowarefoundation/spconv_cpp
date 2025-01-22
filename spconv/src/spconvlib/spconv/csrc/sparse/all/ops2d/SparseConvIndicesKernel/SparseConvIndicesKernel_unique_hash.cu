#include <spconvlib/spconv/csrc/sparse/all/ops2d/SparseConvIndicesKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops2d {
using TensorView = spconvlib::cumm::common::TensorView;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
using TensorViewHashKernel = spconvlib::cumm::common::TensorViewHashKernel;
using ConvLocIter = spconvlib::spconv::csrc::sparse::all::ops2d::spinds::ConvOutLocIter;
using ConvProblem = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::spinds::ConvProblem;
using ConvLocIter64 = spconvlib::spconv::csrc::sparse::all::ops2d::spinds64::ConvOutLocIter;
int SparseConvIndicesKernel::unique_hash(tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor uniq_cnt, tv::Tensor out_indices_offset, int num_out_bound, std::uintptr_t stream_int)   {
  
  auto custream = reinterpret_cast<cudaStream_t>(stream_int);
  tv::cuda::Launch lanucher_build_hash(hashdata_k.size(), custream);
  auto tvctx = tv::Context();
  tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream_int));
  if (num_out_bound <= 0){
      num_out_bound = out_indices_offset.dim(0);
  }
  tv::dispatch<int32_t, int64_t>(hashdata_k.dtype(), [&](auto I){
      using V = int32_t;
      using K = TV_DECLTYPE(I);
      using table_t =
          tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                      tv::hash::default_empty_key_v<K>, false>;
      table_t table = table_t(hashdata_k.data_ptr<K>(), hashdata_v.data_ptr<V>(), hashdata_k.dim(0));
      lanucher_build_hash(arange_hash_table<table_t>, table, 
          out_indices_offset.data_ptr<K>(),
          uniq_cnt.data_ptr<int>(), num_out_bound);
  });
  auto uniq_cnt_cpu = uniq_cnt.cpu(tvctx);
  return std::min(uniq_cnt_cpu.data_ptr<int>()[0], num_out_bound);
}
} // namespace ops2d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib