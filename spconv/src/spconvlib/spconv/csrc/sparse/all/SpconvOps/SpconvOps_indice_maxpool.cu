#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/maxpool/IndiceMaxPool.h>
#include <spconvlib/spconv/csrc/sparse/maxpool/IndiceMaxPoolCPU.h>
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
using IndiceMaxPool = spconvlib::spconv::csrc::sparse::maxpool::IndiceMaxPool;
using IndiceMaxPoolCPU = spconvlib::spconv::csrc::sparse::maxpool::IndiceMaxPoolCPU;
void SpconvOps::indice_maxpool(tv::Tensor out_features, tv::Tensor features, tv::Tensor indice_pairs, tv::Tensor indice_pair_num, int num_activate_out, std::uintptr_t stream)   {
  
  tv::check_shape(out_features, {-1, features.dim(1)});
  auto indice_pair_num_cpu = indice_pair_num.cpu();
  auto indice_pair_num_cpu_ptr = indice_pair_num_cpu.data_ptr<int>();
  for (int i = 0; i < indice_pair_num.dim(0); ++i){
    int nhot = indice_pair_num_cpu_ptr[i];
    nhot = std::min(nhot, int(indice_pairs.dim(2)));
    if (nhot <= 0){
        continue;
    }
    auto inp_indices = indice_pairs[0][i].slice_first_axis(0, nhot);
    auto out_indices = indice_pairs[1][i].slice_first_axis(0, nhot);
    if (features.is_cpu()){
        IndiceMaxPoolCPU::forward(out_features, features, out_indices, inp_indices);
    }
    else{
      IndiceMaxPool::forward(out_features, features, out_indices, inp_indices, stream);
    }
  }
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib