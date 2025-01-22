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
std::unordered_map<std::string, tv::Tensor> SpconvOps::get_indice_gen_tensors_from_workspace(uint8_t* workspace, size_t kv, size_t num_act_in, size_t num_act_out_bound, size_t max_act_out_in_theory, bool subm, bool use_int64_hash_k, bool direct_table)   {
  
  std::unordered_map<std::string, tv::Tensor> res;
  auto ws_prev = workspace;
  auto expected_size = get_indice_gen_workspace_size(kv, num_act_in, num_act_out_bound, 
      max_act_out_in_theory, subm, use_int64_hash_k, direct_table);
  int hash_size = 2 * num_act_out_bound;
  if (direct_table){
      hash_size = tv::align_up(int(1.1 * max_act_out_in_theory), 2);
  }
  if (use_int64_hash_k){
      auto ten = tv::from_blob(workspace, {int64_t(hash_size)}, tv::int64, 0);
      res.insert({"HashKOrKV", ten});
      workspace += ten.nbytes();
      auto ten2 = tv::from_blob(workspace, {int64_t(hash_size)}, tv::int32, 0);
      res.insert({"HashV", ten2});
      workspace += ten2.nbytes();
  }else{
      auto ten = tv::from_blob(workspace, {2, int64_t(hash_size)}, tv::int32, 0);
      res.insert({"HashKOrKV", ten});
      workspace += ten.nbytes();
  }
  if (!subm){
      size_t pair_single_size = kv * int64_t(num_act_in);
      auto ten = tv::from_blob(workspace, {int64_t(pair_single_size + 1)}, use_int64_hash_k ? tv::int64 : tv::int32, 0);
      res.insert({"IndicePairsUniq", ten});
      workspace += ten.nbytes();
      auto ten2 = tv::from_blob(workspace, {int64_t(pair_single_size + 1)}, use_int64_hash_k ? tv::int64 : tv::int32, 0);
      res.insert({"IndicePairsUniqBackup", ten2});
      workspace += ten2.nbytes();
  }
  auto uniq_cnt = tv::from_blob(workspace, {1}, tv::int32, 0);
  res.insert({"TightUniqueCount", uniq_cnt});
  workspace += uniq_cnt.nbytes();
  TV_ASSERT_RT_ERR(workspace - ws_prev == expected_size, "this shouldn't happen", kv, num_act_in,num_act_out_bound,  max_act_out_in_theory,
      subm, use_int64_hash_k, direct_table, "expected", expected_size, workspace - ws_prev);
  return res;
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib