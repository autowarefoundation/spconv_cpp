#include <spconvlib/spconv/csrc/sparse/all/ops4d/SparseConvIndicesKernel.h>
#include <spconvlib/cumm/common/ThrustLib.h>
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
using ThrustLib = spconvlib::cumm::common::ThrustLib;
int SparseConvIndicesKernel::generate_conv_inds_stage1_5(tv::Tensor indice_pairs_uniq, int64_t uniq_size, std::uintptr_t stream_int)   {
  
  int num_out_act = 0;
  tv::dispatch<int32_t, int64_t>(indice_pairs_uniq.dtype(), [&](auto I){
      using T = TV_DECLTYPE(I);
      thrust::device_ptr<T> ptr_tr(indice_pairs_uniq.data_ptr<T>());
      auto thrust_ctx = thrust::cuda::par.on(reinterpret_cast<cudaStream_t>(stream_int));
      thrust::sort(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
      auto new_end = thrust::unique(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
      num_out_act = new_end - ptr_tr - 1;
  });
  return num_out_act;
}
} // namespace ops4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib