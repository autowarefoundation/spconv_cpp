#pragma once
#include <tensorview/hash/ops.h>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/cumm/common/TensorViewHashKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops1d/layout_ns/TensorGeneric.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops1d {
namespace kernel {
using TensorView = spconvlib::cumm::common::TensorView;
using TensorViewHashKernel = spconvlib::cumm::common::TensorViewHashKernel;
using Layout = spconvlib::spconv::csrc::sparse::all::ops1d::layout_ns::TensorGeneric;
template <typename TTable>
__global__ void build_hash_table(TTable table, float const* points, int64_t * points_indice_data, int point_stride, tv::array<float, 1> vsize, tv::array<float, 2> coors_range, tv::array<int, 1> grid_bound, tv::array<int64_t, 1> grid_stride, int num_points)   {
  
  for (int i : tv::KernelLoopX<int>(num_points)){
      bool failed = false;
      int c;
      int64_t prod = 0;
  #pragma unroll
      for (int j = 0; j < 1; ++j) {
          c = floor((points[i * point_stride + 0 - j] - coors_range[j]) /
                      vsize[j]);
          if ((c < 0 || c >= grid_bound[j])) {
              failed = true;
          }
          prod += grid_stride[j] * int64_t(c);
      }
      if (!failed){
          points_indice_data[i] = prod;
          table.insert(prod, i);
      }else{
          points_indice_data[i] = -1;
      }
  }
}
template <typename TTable>
__global__ void assign_table(TTable table, int* indices, int* count, Layout layout, int max_voxels)   {
  
  auto data = table.data();
  for (int i : tv::KernelLoopX<int>(table.size())){
      auto &item = data[i];
      if (!item.empty()) {
          item.second = tv::cuda::atomicAggInc(count);
          if (item.second < max_voxels){
              layout.inverse(item.first, indices + item.second * 1);
          }
      }
  }
}
template <typename TTable>
__global__ void generate_voxel(TTable table, float const* points, const int64_t* points_indice_data, float * voxels, int * num_per_voxel, int64_t* points_voxel_id, int point_stride, int max_points_per_voxel, int max_voxels, tv::array<float, 1> vsize, tv::array<float, 2> coors_range, tv::array<int, 1> grid_bound, tv::array<int64_t, 1> grid_stride, int num_points)   {
  
  int voxel_stride0 = point_stride * max_points_per_voxel;
  for (int i : tv::KernelLoopX<int>(num_points)){
      int64_t prod = points_indice_data[i];
      int voxel_id = -1;
      if (prod != -1){
          auto voxel_index_pair = table.lookup(prod);
          if (!voxel_index_pair.empty() &&
              voxel_index_pair.second < max_voxels) {
              voxel_id = voxel_index_pair.second;
              int old = atomicAdd(num_per_voxel + voxel_index_pair.second, 1);
              if (old < max_points_per_voxel) {
                  for (int j = 0; j < point_stride; ++j) {
                      voxels[voxel_index_pair.second * voxel_stride0 + old * point_stride + j] = points[i * point_stride + j];
                  }
              }
          }
      }
      points_voxel_id[i] = voxel_id;
  }
}
/**
 * @param voxels 
 * @param num_per_voxel 
 * @param num_voxels 
 * @param num_points_per_voxel 
 * @param num_voxel_features 
 */
__global__ void voxel_empty_fill_mean(float * voxels, int * num_per_voxel, int num_voxels, int num_points_per_voxel, int num_voxel_features);
/**
 * @param num_per_voxel 
 * @param num_voxels 
 * @param num_points_per_voxel 
 */
__global__ void limit_num_per_voxel_value(int * num_per_voxel, int num_voxels, int num_points_per_voxel);
struct Point2VoxelKernel {
};
} // namespace kernel
} // namespace ops1d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib