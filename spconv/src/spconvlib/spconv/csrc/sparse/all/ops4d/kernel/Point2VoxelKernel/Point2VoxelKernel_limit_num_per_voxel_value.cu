#include <spconvlib/spconv/csrc/sparse/all/ops4d/kernel/Point2VoxelKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops4d {
namespace kernel {
using TensorView = spconvlib::cumm::common::TensorView;
using TensorViewHashKernel = spconvlib::cumm::common::TensorViewHashKernel;
using Layout = spconvlib::spconv::csrc::sparse::all::ops4d::layout_ns::TensorGeneric;
__global__ void limit_num_per_voxel_value(int * num_per_voxel, int num_voxels, int num_points_per_voxel)   {
  
  for (int i : tv::KernelLoopX<int>(num_voxels)){
      int count = min(num_points_per_voxel, num_per_voxel[i]);
      num_per_voxel[i] = count;
  }
}
} // namespace kernel
} // namespace ops4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib