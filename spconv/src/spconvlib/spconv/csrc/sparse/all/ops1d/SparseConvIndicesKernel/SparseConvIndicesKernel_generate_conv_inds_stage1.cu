#include <spconvlib/spconv/csrc/sparse/all/ops1d/SparseConvIndicesKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops1d {
using TensorView = spconvlib::cumm::common::TensorView;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
using TensorViewHashKernel = spconvlib::cumm::common::TensorViewHashKernel;
using ConvLocIter = spconvlib::spconv::csrc::sparse::all::ops1d::spinds::ConvOutLocIter;
using ConvProblem = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::spinds::ConvProblem;
using ConvLocIter64 = spconvlib::spconv::csrc::sparse::all::ops1d::spinds64::ConvOutLocIter;
void SparseConvIndicesKernel::generate_conv_inds_stage1(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 1> output_dims, tv::array<int, 1> input_dims, tv::array<int, 1> ksize, tv::array<int, 1> stride, tv::array<int, 1> padding, tv::array<int, 1> dilation, bool transposed, std::uintptr_t stream_int)   {
  
  // TODO stream
  // TODO handle num input == 0
  int kv = ksize.op<tv::arrayops::prod>();
  TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");
  // indice_pairs: [2, kv, num_act_in]
  // indice_pairs_uniq: [num_act_in * kv + 1]
  tv::check_shape(indice_pairs, {2, kv, -1});
  // TV_ASSERT_RT_ERR(indice_pairs.dim(-1) == indices.dim(0), "error");
  tv::check_shape(indice_num_per_loc, {kv});
  int64_t uniq_size = indice_pairs.size() / 2 + 1;
  TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) >= uniq_size, "error");
  TV_ASSERT_RT_ERR(indice_num_per_loc.dim(0) == kv, "error");
  tv::cuda::Launch launcher_num_act_in(indices.dim(0), reinterpret_cast<cudaStream_t>(stream_int));
  // tv::cuda::Launch launcher_num_act_in_2(indices.dim(0));
  launcher_num_act_in.blocks.y = kv;
  ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
  bool use_int32 = problem.check_npq_not_overflow();
  tv::cuda::Launch launcher_clean_uniq(uniq_size, reinterpret_cast<cudaStream_t>(stream_int));
  if (int(use_int32) == 0){
    ConvLocIter64 loc_iter(problem);
    tv::dispatch<int32_t, int64_t>(indice_pairs_uniq.dtype(), [&](auto I){
        using T = TV_DECLTYPE(I);
        TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<T>::max(), 
            "kernel volume must smaller than max value of T");
        launcher_clean_uniq(clean_indices_uniq<T>, indice_pairs_uniq.data_ptr<T>(), uniq_size);
        launcher_num_act_in(calc_conv_indices_stage1<T, ConvLocIter64>, loc_iter, indices.data_ptr<const int>(), 
            indice_pairs.data_ptr<int32_t>(), 
            indice_pairs_uniq.data_ptr<T>(), indice_num_per_loc.data_ptr<int>(), indices.dim(0),
            indice_pairs.dim(2), kv, transposed);
    });
  }
  else if (int(use_int32) == 1){
    ConvLocIter loc_iter(problem);
    tv::dispatch<int32_t, int64_t>(indice_pairs_uniq.dtype(), [&](auto I){
        using T = TV_DECLTYPE(I);
        TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<T>::max(), 
            "kernel volume must smaller than max value of T");
        launcher_clean_uniq(clean_indices_uniq<T>, indice_pairs_uniq.data_ptr<T>(), uniq_size);
        launcher_num_act_in(calc_conv_indices_stage1<T, ConvLocIter>, loc_iter, indices.data_ptr<const int>(), 
            indice_pairs.data_ptr<int32_t>(), 
            indice_pairs_uniq.data_ptr<T>(), indice_num_per_loc.data_ptr<int>(), indices.dim(0),
            indice_pairs.dim(2), kv, transposed);
    });
  }
  else{
    TV_THROW_RT_ERR("unknown val int(use_int32), available: [0, 1]")
  }
}
} // namespace ops1d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib