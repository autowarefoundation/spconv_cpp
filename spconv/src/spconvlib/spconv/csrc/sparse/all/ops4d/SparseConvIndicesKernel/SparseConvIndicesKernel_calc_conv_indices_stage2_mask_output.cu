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
__global__ void calc_conv_indices_stage2_mask_output(int* indice_pairs_bwd, uint32_t* mask_bwd, int num_indices_in, int kv, int mask_int_count)   {
  
  for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
      for (int mask_offset = 0; mask_offset < mask_int_count; ++mask_offset){
          uint32_t mask = 0;
          for (int filter_offset = mask_offset * 32; filter_offset < mask_offset * 32 +  32 && filter_offset < kv; ++filter_offset){
              auto val = indice_pairs_bwd[filter_offset * num_indices_in + input_index];
              mask |= (val != -1) << (filter_offset % 32);
          }
          mask_bwd[input_index * mask_int_count + mask_offset] = mask;
      }
  }
}
} // namespace ops4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib