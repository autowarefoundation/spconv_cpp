#pragma once
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu3d/layout_ns/TensorGeneric.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu3d/p2v_c/Point2VoxelCommon.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu3d {
using TensorView = spconvlib::cumm::common::TensorView;
using Layout = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::layout_ns::TensorGeneric;
using Point2VoxelCommon = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::p2v_c::Point2VoxelCommon;
struct Point2VoxelCPU {
  tv::Tensor densehashdata;
  tv::Tensor voxels;
  tv::Tensor indices;
  tv::Tensor num_per_voxel;
  tv::array<float, 3> vsize;
  tv::array<float, 6> coors_range;
  tv::array<int, 3> grid_size;
  tv::array<int, 3> grid_stride;
  
  std::array<int, 3> get_grid_size();
  /**
   * @param vsize_xyz 
   * @param coors_range_xyz 
   */
  static std::tuple<std::array<float, 3>, std::array<int, 3>, std::array<int64_t, 3>, std::array<float, 6>> calc_meta_data(std::array<float, 3> vsize_xyz, std::array<float, 6> coors_range_xyz);
  /**
   * @param vsize_xyz 
   * @param coors_range_xyz 
   * @param num_point_features 
   * @param max_num_voxels 
   * @param max_num_points_per_voxel 
   */
   Point2VoxelCPU(std::array<float, 3> vsize_xyz, std::array<float, 6> coors_range_xyz, int num_point_features, int max_num_voxels, int max_num_points_per_voxel);
  template <typename T, size_t N>
  static tv::array<T, N> array2tvarray(std::array<T, N> arr)   {
    
    tv::array<T, N> tarr;
    for (int i = 0; i < N; ++i){
        tarr[i] = arr[i];
    }
    return tarr;
  }
  template <typename T, size_t N>
  static std::array<T, N> tvarray2array(tv::array<T, N> arr)   {
    
    std::array<T, N> tarr;
    for (int i = 0; i < N; ++i){
        tarr[i] = arr[i];
    }
    return tarr;
  }
  /**
   * @param points 
   * @param voxels 
   * @param indices 
   * @param num_per_voxel 
   * @param densehashdata 
   * @param points_voxel_id 
   * @param vsize 
   * @param grid_size 
   * @param grid_stride 
   * @param coors_range 
   * @param clear_voxels 
   */
  static std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> point_to_voxel_static(tv::Tensor points, tv::Tensor voxels, tv::Tensor indices, tv::Tensor num_per_voxel, tv::Tensor densehashdata, tv::Tensor points_voxel_id, std::array<float, 3> vsize, std::array<int, 3> grid_size, std::array<int, 3> grid_stride, std::array<float, 6> coors_range, bool clear_voxels = true);
  /**
   * @param points 
   * @param voxels 
   * @param indices 
   * @param num_per_voxel 
   * @param densehashdata 
   * @param points_voxel_id 
   * @param vsize 
   * @param grid_size 
   * @param grid_stride 
   * @param coors_range 
   * @param clear_voxels 
   */
  static std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> point_to_voxel_empty_mean_static(tv::Tensor points, tv::Tensor voxels, tv::Tensor indices, tv::Tensor num_per_voxel, tv::Tensor densehashdata, tv::Tensor points_voxel_id, std::array<float, 3> vsize, std::array<int, 3> grid_size, std::array<int, 3> grid_stride, std::array<float, 6> coors_range, bool clear_voxels = true);
  /**
   * @param points 
   * @param clear_voxels 
   */
  std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> point_to_voxel(tv::Tensor points, bool clear_voxels = true);
  /**
   * @param points 
   * @param clear_voxels 
   */
  std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> point_to_voxel_empty_mean(tv::Tensor points, bool clear_voxels = true);
};
} // namespace ops_cpu3d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib