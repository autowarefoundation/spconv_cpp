#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/all/ops1d/SparseConvIndicesKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops2d/SparseConvIndicesKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops3d/SparseConvIndicesKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops4d/SparseConvIndicesKernel.h>
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
using SpconvIndices1D = spconvlib::spconv::csrc::sparse::all::ops1d::SparseConvIndicesKernel;
using SpconvIndices2D = spconvlib::spconv::csrc::sparse::all::ops2d::SparseConvIndicesKernel;
using SpconvIndices3D = spconvlib::spconv::csrc::sparse::all::ops3d::SparseConvIndicesKernel;
using SpconvIndices4D = spconvlib::spconv::csrc::sparse::all::ops4d::SparseConvIndicesKernel;
int SpconvOps::unique_hash(tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor uniq_cnt, tv::Tensor out_indices_offset, int num_out_bound, std::uintptr_t stream_int)   {
  
  return SpconvIndices3D::unique_hash(hashdata_k, hashdata_v, 
      uniq_cnt, out_indices_offset, num_out_bound, stream_int);
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib