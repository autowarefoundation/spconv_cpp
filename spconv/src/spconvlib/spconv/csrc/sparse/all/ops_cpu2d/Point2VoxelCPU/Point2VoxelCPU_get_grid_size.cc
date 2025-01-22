#include <spconvlib/spconv/csrc/sparse/all/ops_cpu2d/Point2VoxelCPU.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu2d {
using TensorView = spconvlib::cumm::common::TensorView;
using Layout = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::layout_ns::TensorGeneric;
using Point2VoxelCommon = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::p2v_c::Point2VoxelCommon;
std::array<int, 2> Point2VoxelCPU::get_grid_size()   {
  
  std::array<int, 2> res;
  for (int i = 0; i < 2; ++i){
      res[i] = grid_size[i];
  }
  return res;
}
} // namespace ops_cpu2d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib