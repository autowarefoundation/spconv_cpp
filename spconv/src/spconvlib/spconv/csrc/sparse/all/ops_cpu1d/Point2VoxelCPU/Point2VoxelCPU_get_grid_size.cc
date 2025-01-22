#include <spconvlib/spconv/csrc/sparse/all/ops_cpu1d/Point2VoxelCPU.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu1d {
using TensorView = spconvlib::cumm::common::TensorView;
using Layout = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::layout_ns::TensorGeneric;
using Point2VoxelCommon = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::p2v_c::Point2VoxelCommon;
std::array<int, 1> Point2VoxelCPU::get_grid_size()   {
  
  std::array<int, 1> res;
  for (int i = 0; i < 1; ++i){
      res[i] = grid_size[i];
  }
  return res;
}
} // namespace ops_cpu1d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib