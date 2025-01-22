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
std::size_t SpconvOps::get_indice_gen_workspace_size(size_t kv, size_t num_act_in, size_t num_act_out_bound, size_t max_act_out_in_theory, bool subm, bool use_int64_hash_k, bool direct_table)   {
  
  int hash_size = 2 * num_act_out_bound;
  if (direct_table){
      hash_size = tv::align_up(int(1.1 * max_act_out_in_theory), 2);
  }
  size_t res = 0;
  if (subm){
      res = hash_size * (use_int64_hash_k ? 3 : 2) * sizeof(int) + 1 * sizeof(int);
  }else{
      size_t pair_single_size = kv * num_act_in; // 40000
      size_t ind_uniq_and_bkp_size = (pair_single_size + 1) * 2 * (use_int64_hash_k ? sizeof(int64_t) : sizeof(int32_t));
      hash_size = hash_size * (use_int64_hash_k ? 3 : 2) * sizeof(int);
      res = ind_uniq_and_bkp_size + hash_size + 1 * sizeof(int);
  }
  return res;
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib