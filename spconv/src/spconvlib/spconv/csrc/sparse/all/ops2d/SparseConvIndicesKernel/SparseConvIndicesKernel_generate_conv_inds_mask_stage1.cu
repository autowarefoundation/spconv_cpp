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
void SparseConvIndicesKernel::generate_conv_inds_mask_stage1(tv::Tensor indices, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 2> output_dims, tv::array<int, 2> input_dims, tv::array<int, 2> ksize, tv::array<int, 2> stride, tv::array<int, 2> padding, tv::array<int, 2> dilation, bool transposed, std::uintptr_t stream_int)   {
  
  // TODO stream
  // TODO handle num input == 0
  int kv = ksize.op<tv::arrayops::prod>();
  int num_act_in = indices.dim(0);
  // indice_pairs_bwd: [kv, num_act_in] or empty
  // indice_pairs_uniq: [kv * num_act_in + 1]
  if (!indice_pairs_bwd.empty()){
      tv::check_shape(indice_pairs_bwd, {kv, num_act_in});
  }
  tv::check_shape(indice_num_per_loc, {kv});
  int64_t uniq_size = kv * num_act_in + 1;
  TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) == uniq_size, "error");
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
        launcher_num_act_in(calc_conv_indices_stage1_mask<T, ConvLocIter64>, loc_iter, indices.data_ptr<const int>(), 
            indice_pairs_bwd.data_ptr<int32_t>(), 
            indice_pairs_uniq.data_ptr<T>(), indice_num_per_loc.data_ptr<int>(), indices.dim(0),
            kv, transposed);
    });
  }
  else if (int(use_int32) == 1){
    ConvLocIter loc_iter(problem);
    tv::dispatch<int32_t, int64_t>(indice_pairs_uniq.dtype(), [&](auto I){
        using T = TV_DECLTYPE(I);
        TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<T>::max(), 
            "kernel volume must smaller than max value of T");
        launcher_clean_uniq(clean_indices_uniq<T>, indice_pairs_uniq.data_ptr<T>(), uniq_size);
        launcher_num_act_in(calc_conv_indices_stage1_mask<T, ConvLocIter>, loc_iter, indices.data_ptr<const int>(), 
            indice_pairs_bwd.data_ptr<int32_t>(), 
            indice_pairs_uniq.data_ptr<T>(), indice_num_per_loc.data_ptr<int>(), indices.dim(0),
            kv, transposed);
    });
  }
  else{
    TV_THROW_RT_ERR("unknown val int(use_int32), available: [0, 1]")
  }
}
} // namespace ops2d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib