#include <spconvlib/spconv/csrc/sparse/all/ops3d/kernel/Point2VoxelKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {
namespace kernel {
using TensorView = spconvlib::cumm::common::TensorView;
using TensorViewHashKernel = spconvlib::cumm::common::TensorViewHashKernel;
using Layout = spconvlib::spconv::csrc::sparse::all::ops3d::layout_ns::TensorGeneric;
__global__ void voxel_empty_fill_mean(float * voxels, int * num_per_voxel, int num_voxels, int num_points_per_voxel, int num_voxel_features)   {
  
  int voxel_stride = num_points_per_voxel * num_voxel_features;
  for (int i : tv::KernelLoopX<int>(num_voxels)){
      int count = min(num_points_per_voxel, num_per_voxel[i]);
      num_per_voxel[i] = count;
      for (int j = 0; j < num_voxel_features; ++j){
          auto voxel_ptr = voxels + i * voxel_stride + j;
          float sum_val = 0;
          for (int k = 0; k < count; ++k){
              sum_val += voxel_ptr[0];
              voxel_ptr += num_voxel_features;
          }
          sum_val = count == 0 ? 0 : sum_val / count;
          for (int k = count; k < num_points_per_voxel; ++k){
              voxel_ptr[0] = sum_val;
              voxel_ptr += num_voxel_features;
          }
      }
  }
}
} // namespace kernel
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib