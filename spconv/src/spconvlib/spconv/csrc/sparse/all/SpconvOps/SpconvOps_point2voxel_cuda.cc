#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
using ThrustCustomAllocatorV2 = spconvlib::spconv::csrc::sparse::all::ThrustCustomAllocatorV2;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using ThrustAllocator = spconvlib::spconv::csrc::sparse::alloc::ThrustAllocator;
using Point2Voxel1DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::Point2VoxelCPU;
using SpconvIndicesCPU1D = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::SparseConvIndicesCPU;
using Point2Voxel1D = spconvlib::spconv::csrc::sparse::all::ops1d::Point2Voxel;
using Point2Voxel2DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::Point2VoxelCPU;
using SpconvIndicesCPU2D = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::SparseConvIndicesCPU;
using Point2Voxel2D = spconvlib::spconv::csrc::sparse::all::ops2d::Point2Voxel;
using Point2Voxel3DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::Point2VoxelCPU;
using SpconvIndicesCPU3D = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::SparseConvIndicesCPU;
using Point2Voxel3D = spconvlib::spconv::csrc::sparse::all::ops3d::Point2Voxel;
using Point2Voxel4DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::Point2VoxelCPU;
using SpconvIndicesCPU4D = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::SparseConvIndicesCPU;
using Point2Voxel4D = spconvlib::spconv::csrc::sparse::all::ops4d::Point2Voxel;
std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> SpconvOps::point2voxel_cuda(tv::Tensor points, tv::Tensor voxels, tv::Tensor indices, tv::Tensor num_per_voxel, tv::Tensor hashdata, tv::Tensor point_indice_data, tv::Tensor pc_voxel_id, std::vector<float> vsize, std::vector<int> grid_size, std::vector<int64_t> grid_stride, std::vector<float> coors_range, bool empty_mean, bool clear_voxels, std::uintptr_t stream_int)   {
  
  int ndim = vsize.size();
  TV_ASSERT_RT_ERR(vsize.size() == ndim && grid_stride.size() == ndim && 
      coors_range.size() == ndim * 2 && grid_size.size() == ndim, 
      "your params size not equal to ndim", ndim);
  // voxels: []
  if (ndim == 1){
      std::array<float, 1> vsize_;
      std::array<int, 1> grid_size_;
      std::array<int64_t, 1> grid_stride_;
      std::array<float, 2> coors_range_;
      for (int i = 0; i < 1; ++i){
          vsize_[i] = vsize[i];
          grid_size_[i] = grid_size[i];
          grid_stride_[i] = grid_stride[i];
          coors_range_[i] = coors_range[i];
          coors_range_[i + 1] = coors_range[i + 1];
      }
      return Point2Voxel1D::point_to_voxel_hash_static(points, voxels, indices, 
          num_per_voxel, hashdata, point_indice_data, pc_voxel_id,
          vsize_, grid_size_, grid_stride_, coors_range_, clear_voxels, 
          empty_mean, stream_int);
  }
  if (ndim == 2){
      std::array<float, 2> vsize_;
      std::array<int, 2> grid_size_;
      std::array<int64_t, 2> grid_stride_;
      std::array<float, 4> coors_range_;
      for (int i = 0; i < 2; ++i){
          vsize_[i] = vsize[i];
          grid_size_[i] = grid_size[i];
          grid_stride_[i] = grid_stride[i];
          coors_range_[i] = coors_range[i];
          coors_range_[i + 2] = coors_range[i + 2];
      }
      return Point2Voxel2D::point_to_voxel_hash_static(points, voxels, indices, 
          num_per_voxel, hashdata, point_indice_data, pc_voxel_id,
          vsize_, grid_size_, grid_stride_, coors_range_, clear_voxels, 
          empty_mean, stream_int);
  }
  if (ndim == 3){
      std::array<float, 3> vsize_;
      std::array<int, 3> grid_size_;
      std::array<int64_t, 3> grid_stride_;
      std::array<float, 6> coors_range_;
      for (int i = 0; i < 3; ++i){
          vsize_[i] = vsize[i];
          grid_size_[i] = grid_size[i];
          grid_stride_[i] = grid_stride[i];
          coors_range_[i] = coors_range[i];
          coors_range_[i + 3] = coors_range[i + 3];
      }
      return Point2Voxel3D::point_to_voxel_hash_static(points, voxels, indices, 
          num_per_voxel, hashdata, point_indice_data, pc_voxel_id,
          vsize_, grid_size_, grid_stride_, coors_range_, clear_voxels, 
          empty_mean, stream_int);
  }
  if (ndim == 4){
      std::array<float, 4> vsize_;
      std::array<int, 4> grid_size_;
      std::array<int64_t, 4> grid_stride_;
      std::array<float, 8> coors_range_;
      for (int i = 0; i < 4; ++i){
          vsize_[i] = vsize[i];
          grid_size_[i] = grid_size[i];
          grid_stride_[i] = grid_stride[i];
          coors_range_[i] = coors_range[i];
          coors_range_[i + 4] = coors_range[i + 4];
      }
      return Point2Voxel4D::point_to_voxel_hash_static(points, voxels, indices, 
          num_per_voxel, hashdata, point_indice_data, pc_voxel_id,
          vsize_, grid_size_, grid_stride_, coors_range_, clear_voxels, 
          empty_mean, stream_int);
  }
  TV_THROW_RT_ERR("unknown ndim", ndim);
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib