#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/all/CustomThrustLib.h>
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
using CustomThrustLib = spconvlib::spconv::csrc::sparse::all::CustomThrustLib;
int SpconvOps::apply_thrust_unique_to_indice_pairs_uniq(tv::Tensor data, ThrustAllocator& allocator, std::uintptr_t stream_int)   {
  
  int num_out_act = 0;
  int uniq_size = data.dim(0);
  tv::dispatch<int32_t, int64_t>(data.dtype(), [&](auto I){
      using T = TV_DECLTYPE(I);
      thrust::device_ptr<T> ptr_tr(data.data_ptr<T>());
      auto thrust_ctx = thrust::cuda::par(allocator).on(reinterpret_cast<cudaStream_t>(stream_int));
      thrust::sort(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
      auto new_end = thrust::unique(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
      num_out_act = new_end - ptr_tr - 1;
  });
  return num_out_act;
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib