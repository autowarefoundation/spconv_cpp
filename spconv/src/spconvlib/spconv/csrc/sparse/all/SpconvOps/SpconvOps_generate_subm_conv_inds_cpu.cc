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
int SpconvOps::generate_subm_conv_inds_cpu(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> dilation)   {
  
  int ndim = indices.dim(1) - 1;
  TV_ASSERT_RT_ERR(input_dims.size() == ndim &&
      ksize.size() == ndim && dilation.size() == ndim, "your params size not equal to ndim", ndim);
  if (ndim == 1){
      tv::array<int, 1> input_dims_;
      tv::array<int, 1> ksize_, dilation_;
      for (int i = 0; i < 1; ++i){
          input_dims_[i] = input_dims[i];
          ksize_[i] = ksize[i];
          dilation_[i] = dilation[i];
      }
      return SpconvIndicesCPU1D::generate_subm_conv_inds(indices,
          indice_pairs, out_inds, indice_num_per_loc,
          batch_size, input_dims_, 
          ksize_, dilation_);
  }
  if (ndim == 2){
      tv::array<int, 2> input_dims_;
      tv::array<int, 2> ksize_, dilation_;
      for (int i = 0; i < 2; ++i){
          input_dims_[i] = input_dims[i];
          ksize_[i] = ksize[i];
          dilation_[i] = dilation[i];
      }
      return SpconvIndicesCPU2D::generate_subm_conv_inds(indices,
          indice_pairs, out_inds, indice_num_per_loc,
          batch_size, input_dims_, 
          ksize_, dilation_);
  }
  if (ndim == 3){
      tv::array<int, 3> input_dims_;
      tv::array<int, 3> ksize_, dilation_;
      for (int i = 0; i < 3; ++i){
          input_dims_[i] = input_dims[i];
          ksize_[i] = ksize[i];
          dilation_[i] = dilation[i];
      }
      return SpconvIndicesCPU3D::generate_subm_conv_inds(indices,
          indice_pairs, out_inds, indice_num_per_loc,
          batch_size, input_dims_, 
          ksize_, dilation_);
  }
  if (ndim == 4){
      tv::array<int, 4> input_dims_;
      tv::array<int, 4> ksize_, dilation_;
      for (int i = 0; i < 4; ++i){
          input_dims_[i] = input_dims[i];
          ksize_[i] = ksize[i];
          dilation_[i] = dilation[i];
      }
      return SpconvIndicesCPU4D::generate_subm_conv_inds(indices,
          indice_pairs, out_inds, indice_num_per_loc,
          batch_size, input_dims_, 
          ksize_, dilation_);
  }
  TV_THROW_RT_ERR("unknown ndim", ndim);
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib