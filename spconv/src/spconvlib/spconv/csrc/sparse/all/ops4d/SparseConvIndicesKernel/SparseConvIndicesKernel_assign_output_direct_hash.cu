#include <spconvlib/spconv/csrc/sparse/all/ops4d/SparseConvIndicesKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops4d {
using TensorView = spconvlib::cumm::common::TensorView;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
using TensorViewHashKernel = spconvlib::cumm::common::TensorViewHashKernel;
using ConvLocIter = spconvlib::spconv::csrc::sparse::all::ops4d::spinds::ConvOutLocIter;
using ConvProblem = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::spinds::ConvProblem;
using ConvLocIter64 = spconvlib::spconv::csrc::sparse::all::ops4d::spinds64::ConvOutLocIter;
void SparseConvIndicesKernel::assign_output_direct_hash(tv::Tensor out_indices_offset, tv::Tensor out_inds, int batch_size, tv::array<int, 4> output_dims, tv::array<int, 4> input_dims, tv::array<int, 4> ksize, tv::array<int, 4> stride, tv::array<int, 4> padding, tv::array<int, 4> dilation, std::uintptr_t stream_int)   {
  
  auto custream = reinterpret_cast<cudaStream_t>(stream_int);
  tv::cuda::Launch lanucher_build_hash(out_inds.dim(0), custream);
  TV_ASSERT_RT_ERR(out_indices_offset.dim(0) >= out_inds.dim(0), "error");
  ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
  bool use_int32 = problem.check_npq_not_overflow();
  auto tvctx = tv::Context();
  tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream_int));
  if (int(use_int32) == 0){
    ConvLocIter64 loc_iter(problem);
    tv::dispatch<int32_t, int64_t>(out_indices_offset.dtype(), [&](auto I){
        using K = TV_DECLTYPE(I);
        lanucher_build_hash(assign_out_indices<K, std::decay_t<decltype(loc_iter.layout_npq)>>, out_inds.data_ptr<int>(),
            out_indices_offset.data_ptr<const K>(),
            loc_iter.layout_npq, out_inds.dim(0));
    });
  }
  else if (int(use_int32) == 1){
    ConvLocIter loc_iter(problem);
    tv::dispatch<int32_t, int64_t>(out_indices_offset.dtype(), [&](auto I){
        using K = TV_DECLTYPE(I);
        lanucher_build_hash(assign_out_indices<K, std::decay_t<decltype(loc_iter.layout_npq)>>, out_inds.data_ptr<int>(),
            out_indices_offset.data_ptr<const K>(),
            loc_iter.layout_npq, out_inds.dim(0));
    });
  }
  else{
    TV_THROW_RT_ERR("unknown val int(use_int32), available: [0, 1]")
  }
}
} // namespace ops4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib