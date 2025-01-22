#include <spconvlib/spconv/csrc/sparse/all/ops4d/p2v_c/Point2VoxelCommon.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops4d {
namespace p2v_c {
using TensorView = spconvlib::cumm::common::TensorView;
std::tuple<std::array<float, 4>, std::array<int, 4>, std::array<int64_t, 4>, std::array<float, 8>> Point2VoxelCommon::calc_meta_data(std::array<float, 4> vsize_xyz, std::array<float, 8> coors_range_xyz)   {
  
  std::array<float, 4> vsize;
  std::array<int, 4> grid_size;
  std::array<int64_t, 4> grid_stride;
  std::array<float, 8> coors_range;
  for (int i = 0; i < 4; ++i){
      vsize[3 - i] = vsize_xyz[i];
      coors_range[3 - i] = coors_range_xyz[i];
      coors_range[7 - i] = coors_range_xyz[i + 4];
  }
  int64_t prod = 1;
  for (size_t i = 0; i < 4; ++i) {
      grid_size[i] =
          std::round((coors_range[4 + i] - coors_range[i]) / vsize[i]);
  }
  for (int i = 4 - 1; i >= 0; --i) {
      grid_stride[i] = prod;
      prod *= grid_size[i];
  }
  return std::make_tuple(vsize, grid_size, grid_stride, coors_range);
}
} // namespace p2v_c
} // namespace ops4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib