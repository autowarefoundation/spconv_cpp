#include <spconvlib/spconv/csrc/sparse/all/ops_cpu4d/Point2VoxelCPU.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu4d {
using TensorView = spconvlib::cumm::common::TensorView;
using Layout = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::layout_ns::TensorGeneric;
using Point2VoxelCommon = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::p2v_c::Point2VoxelCommon;
std::array<int, 4> Point2VoxelCPU::get_grid_size()   {
  
  std::array<int, 4> res;
  for (int i = 0; i < 4; ++i){
      res[i] = grid_size[i];
  }
  return res;
}
} // namespace ops_cpu4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib