#include <spconvlib/spconv/csrc/sparse/all/ops3d/p2v_c/Point2VoxelCommon.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {
namespace p2v_c {
using TensorView = spconvlib::cumm::common::TensorView;
std::tuple<std::array<float, 3>, std::array<int, 3>, std::array<int64_t, 3>, std::array<float, 6>> Point2VoxelCommon::calc_meta_data(std::array<float, 3> vsize_xyz, std::array<float, 6> coors_range_xyz)   {
  
  std::array<float, 3> vsize;
  std::array<int, 3> grid_size;
  std::array<int64_t, 3> grid_stride;
  std::array<float, 6> coors_range;
  for (int i = 0; i < 3; ++i){
      vsize[2 - i] = vsize_xyz[i];
      coors_range[2 - i] = coors_range_xyz[i];
      coors_range[5 - i] = coors_range_xyz[i + 3];
  }
  int64_t prod = 1;
  for (size_t i = 0; i < 3; ++i) {
      grid_size[i] =
          std::round((coors_range[3 + i] - coors_range[i]) / vsize[i]);
  }
  for (int i = 3 - 1; i >= 0; --i) {
      grid_stride[i] = prod;
      prod *= grid_size[i];
  }
  return std::make_tuple(vsize, grid_size, grid_stride, coors_range);
}
} // namespace p2v_c
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib