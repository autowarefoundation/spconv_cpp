#include <spconvlib/spconv/csrc/sparse/all/ops1d/Point2Voxel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops1d {
using TensorView = spconvlib::cumm::common::TensorView;
using Point2VoxelCommon = spconvlib::spconv::csrc::sparse::all::ops1d::p2v_c::Point2VoxelCommon;
using Layout = spconvlib::spconv::csrc::sparse::all::ops1d::layout_ns::TensorGeneric;
std::array<int, 1> Point2Voxel::get_grid_size()   {
  
  std::array<int, 1> res;
  for (int i = 0; i < 1; ++i){
      res[i] = grid_size[i];
  }
  return res;
}
} // namespace ops1d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib