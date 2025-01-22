#include <spconvlib/spconv/csrc/sparse/all/ops_cpu2d/Point2VoxelCPU.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu2d {
using TensorView = spconvlib::cumm::common::TensorView;
using Layout = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::layout_ns::TensorGeneric;
using Point2VoxelCommon = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::p2v_c::Point2VoxelCommon;
std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> Point2VoxelCPU::point_to_voxel_empty_mean_static(tv::Tensor points, tv::Tensor voxels, tv::Tensor indices, tv::Tensor num_per_voxel, tv::Tensor densehashdata, tv::Tensor points_voxel_id, std::array<float, 2> vsize, std::array<int, 2> grid_size, std::array<int, 2> grid_stride, std::array<float, 4> coors_range, bool clear_voxels)   {
  
  auto max_num_voxels = voxels.dim(0);
  auto max_num_points_per_voxel = voxels.dim(1);
  num_per_voxel.zero_();
  if (clear_voxels){
      voxels.zero_();
  }
  auto points_voxel_id_ptr = points_voxel_id.data_ptr<int64_t>();
  int res_voxel_num = 0;
  int num_features = points.dim(1);
  auto N = points.dim(0);
  int c;
  TV_ASSERT_RT_ERR(num_features == voxels.dim(2), "your points num features doesn't equal to voxel.");
  tv::dispatch<float, double>(points.dtype(), [&](auto I){
      using T = decltype(I);
      auto points_rw = points.tview<T, 2>();
      auto coors_rw = indices.tview<int, 2>();
      auto voxels_rw = voxels.tview<float, 3>();
      auto num_points_per_voxel_rw = num_per_voxel.tview<int, 1>();
      int coor[2];
      auto coor_to_voxelidx_rw = densehashdata.tview<int, 2>();
      int voxelidx, num;
      bool failed;
      int voxel_num = 0;
      for (int i = 0; i < N; ++i) {
          failed = false;
          for (int j = 0; j < 2; ++j) {
              c = floor((points_rw(i, 1 - j) - coors_range[j]) / vsize[j]);
              if ((c < 0 || c >= grid_size[j])) {
                  failed = true;
                  break;
              }
              coor[j] = c;
          }
          if (failed){
              points_voxel_id_ptr[i] = -1;
              continue;
          }
          voxelidx = coor_to_voxelidx_rw(coor[0], coor[1]);
          if (voxelidx == -1) {
              voxelidx = voxel_num;
              if (voxel_num >= max_num_voxels){
                  points_voxel_id_ptr[i] = -1;
                  continue;
              }
              voxel_num += 1;
              coor_to_voxelidx_rw(coor[0], coor[1]) = voxelidx;
              for (int k = 0; k < 2; ++k) {
                  coors_rw(voxelidx, k) = coor[k];
              }
          }
          points_voxel_id_ptr[i] = voxelidx;
          num = num_points_per_voxel_rw(voxelidx);
          if (num < max_num_points_per_voxel) {
              // voxel_point_mask_rw(voxelidx, num) = float(1);
              for (int k = 0; k < num_features; ++k) {
                  voxels_rw(voxelidx, num, k) = points_rw(i, k);
              }
              num_points_per_voxel_rw(voxelidx) += 1;
          }
      }
      std::vector<float> mean_value(num_features);
      for (int i = 0; i < voxel_num; ++i) {
          coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1)) = -1;
          if TV_IF_CONSTEXPR (true){
              num = num_points_per_voxel_rw(i);
              if (num > 0){
                  mean_value.clear();
                  for (int j = 0; j < num; ++j) {
                      for (int k = 0; k < num_features; ++k) {
                          mean_value[k] += voxels_rw(i, j, k);
                      }
                  }
                  for (int k = 0; k < num_features; ++k){
                      mean_value[k] /= num;
                  }
                  for (int j = num; j < max_num_points_per_voxel; ++j) {
                      for (int k = 0; k < num_features; ++k) {
                          voxels_rw(i, j, k) = mean_value[k];
                      }
                  }
              }
          }
      }
      res_voxel_num = voxel_num;
  });
  return std::make_tuple(voxels.slice_first_axis(0, res_voxel_num), 
      indices.slice_first_axis(0, res_voxel_num), 
      num_per_voxel.slice_first_axis(0, res_voxel_num));
}
} // namespace ops_cpu2d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib