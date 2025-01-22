#include <spconvlib/spconv/csrc/sparse/all/ops_cpu3d/Point2VoxelCPU.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu3d {
using TensorView = spconvlib::cumm::common::TensorView;
using Layout = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::layout_ns::TensorGeneric;
using Point2VoxelCommon = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::p2v_c::Point2VoxelCommon;
std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> Point2VoxelCPU::point_to_voxel_empty_mean(tv::Tensor points, bool clear_voxels)   {
  
  tv::Tensor points_voxel_id = tv::empty({points.dim(0)}, tv::int64, -1);
  return point_to_voxel_empty_mean_static(points, voxels, indices, num_per_voxel, 
      densehashdata, points_voxel_id, tvarray2array(vsize), 
      tvarray2array(grid_size), tvarray2array(grid_stride), 
      tvarray2array(coors_range), clear_voxels);
}
} // namespace ops_cpu3d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib