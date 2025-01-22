#include <spconvlib/spconv/csrc/sparse/all/ops2d/Point2Voxel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops2d {
using TensorView = spconvlib::cumm::common::TensorView;
using Point2VoxelCommon = spconvlib::spconv::csrc::sparse::all::ops2d::p2v_c::Point2VoxelCommon;
using Layout = spconvlib::spconv::csrc::sparse::all::ops2d::layout_ns::TensorGeneric;
std::array<int, 2> Point2Voxel::get_grid_size()   {
  
  std::array<int, 2> res;
  for (int i = 0; i < 2; ++i){
      res[i] = grid_size[i];
  }
  return res;
}
} // namespace ops2d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib