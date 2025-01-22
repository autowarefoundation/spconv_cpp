#include <spconvlib/spconv/csrc/sparse/all/ops1d/Point2Voxel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops1d/kernel/Point2VoxelKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops1d {
using TensorView = spconvlib::cumm::common::TensorView;
using Point2VoxelCommon = spconvlib::spconv::csrc::sparse::all::ops1d::p2v_c::Point2VoxelCommon;
using Layout = spconvlib::spconv::csrc::sparse::all::ops1d::layout_ns::TensorGeneric;
std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> Point2Voxel::point_to_voxel_hash(tv::Tensor points, bool clear_voxels, bool empty_mean, std::uintptr_t stream_int)   {
  
  tv::Tensor points_voxel_id = tv::empty({points.dim(0)}, tv::int64, 0);
  int64_t expected_hash_data_num = points.dim(0) * 2;
  if (hashdata.dim(0) < expected_hash_data_num){
      hashdata = tv::zeros({expected_hash_data_num}, tv::custom128, 0);
  }
  if (point_indice_data.dim(0) < points.dim(0)){
      point_indice_data = tv::zeros({points.dim(0)}, tv::int64, 0);
  }
  return point_to_voxel_hash_static(points, voxels, indices, num_per_voxel, 
      hashdata, point_indice_data, points_voxel_id, Point2VoxelCommon::tvarray2array(vsize), 
      Point2VoxelCommon::tvarray2array(grid_size), Point2VoxelCommon::tvarray2array(grid_stride), 
      Point2VoxelCommon::tvarray2array(coors_range), clear_voxels, empty_mean, stream_int);
}
} // namespace ops1d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib