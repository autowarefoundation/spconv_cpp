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
std::tuple<std::vector<float>, std::vector<int>, std::vector<int>, std::vector<float>> SpconvOps::calc_point2voxel_meta_data(std::vector<float> vsize_xyz, std::vector<float> coors_range_xyz)   {
  
  int ndim = vsize_xyz.size();
  TV_ASSERT_RT_ERR(vsize_xyz.size() == ndim &&
      coors_range_xyz.size() == ndim * 2, "your params size not equal to ndim", ndim);
  if (ndim == 1){
      std::array<float, 1> vsize_xyz_;
      std::array<float, 2> coors_range_xyz_;
      for (int i = 0; i < 1; ++i){
          vsize_xyz_[i] = vsize_xyz[i];
          coors_range_xyz_[i] = coors_range_xyz[i];
          coors_range_xyz_[i + 1] = coors_range_xyz[i + 1];
      }
      auto res = Point2Voxel1DCPU::calc_meta_data(vsize_xyz_, coors_range_xyz_);
      std::vector<float> vsize(1), coors_range(2);
      std::vector<int> grid_size(1), grid_stride(1);
      for (int i = 0; i < 1; ++i){
          vsize[i] = std::get<0>(res)[i];
          grid_size[i] = std::get<1>(res)[i];
          grid_stride[i] = std::get<2>(res)[i];
          coors_range[i] = std::get<3>(res)[i];
          coors_range[i + 1] = std::get<3>(res)[i + 1];
      }
      return std::make_tuple(vsize, grid_size, grid_stride, coors_range);
  }
  if (ndim == 2){
      std::array<float, 2> vsize_xyz_;
      std::array<float, 4> coors_range_xyz_;
      for (int i = 0; i < 2; ++i){
          vsize_xyz_[i] = vsize_xyz[i];
          coors_range_xyz_[i] = coors_range_xyz[i];
          coors_range_xyz_[i + 2] = coors_range_xyz[i + 2];
      }
      auto res = Point2Voxel2DCPU::calc_meta_data(vsize_xyz_, coors_range_xyz_);
      std::vector<float> vsize(2), coors_range(4);
      std::vector<int> grid_size(2), grid_stride(2);
      for (int i = 0; i < 2; ++i){
          vsize[i] = std::get<0>(res)[i];
          grid_size[i] = std::get<1>(res)[i];
          grid_stride[i] = std::get<2>(res)[i];
          coors_range[i] = std::get<3>(res)[i];
          coors_range[i + 2] = std::get<3>(res)[i + 2];
      }
      return std::make_tuple(vsize, grid_size, grid_stride, coors_range);
  }
  if (ndim == 3){
      std::array<float, 3> vsize_xyz_;
      std::array<float, 6> coors_range_xyz_;
      for (int i = 0; i < 3; ++i){
          vsize_xyz_[i] = vsize_xyz[i];
          coors_range_xyz_[i] = coors_range_xyz[i];
          coors_range_xyz_[i + 3] = coors_range_xyz[i + 3];
      }
      auto res = Point2Voxel3DCPU::calc_meta_data(vsize_xyz_, coors_range_xyz_);
      std::vector<float> vsize(3), coors_range(6);
      std::vector<int> grid_size(3), grid_stride(3);
      for (int i = 0; i < 3; ++i){
          vsize[i] = std::get<0>(res)[i];
          grid_size[i] = std::get<1>(res)[i];
          grid_stride[i] = std::get<2>(res)[i];
          coors_range[i] = std::get<3>(res)[i];
          coors_range[i + 3] = std::get<3>(res)[i + 3];
      }
      return std::make_tuple(vsize, grid_size, grid_stride, coors_range);
  }
  if (ndim == 4){
      std::array<float, 4> vsize_xyz_;
      std::array<float, 8> coors_range_xyz_;
      for (int i = 0; i < 4; ++i){
          vsize_xyz_[i] = vsize_xyz[i];
          coors_range_xyz_[i] = coors_range_xyz[i];
          coors_range_xyz_[i + 4] = coors_range_xyz[i + 4];
      }
      auto res = Point2Voxel4DCPU::calc_meta_data(vsize_xyz_, coors_range_xyz_);
      std::vector<float> vsize(4), coors_range(8);
      std::vector<int> grid_size(4), grid_stride(4);
      for (int i = 0; i < 4; ++i){
          vsize[i] = std::get<0>(res)[i];
          grid_size[i] = std::get<1>(res)[i];
          grid_stride[i] = std::get<2>(res)[i];
          coors_range[i] = std::get<3>(res)[i];
          coors_range[i + 4] = std::get<3>(res)[i + 4];
      }
      return std::make_tuple(vsize, grid_size, grid_stride, coors_range);
  }
  TV_THROW_RT_ERR("unknown ndim", ndim);
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib