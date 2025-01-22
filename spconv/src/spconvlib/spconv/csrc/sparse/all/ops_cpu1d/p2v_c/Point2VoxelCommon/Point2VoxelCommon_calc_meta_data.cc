#include <spconvlib/spconv/csrc/sparse/all/ops_cpu1d/p2v_c/Point2VoxelCommon.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu1d {
namespace p2v_c {
using TensorView = spconvlib::cumm::common::TensorView;
std::tuple<std::array<float, 1>, std::array<int, 1>, std::array<int64_t, 1>, std::array<float, 2>> Point2VoxelCommon::calc_meta_data(std::array<float, 1> vsize_xyz, std::array<float, 2> coors_range_xyz)   {
  
  std::array<float, 1> vsize;
  std::array<int, 1> grid_size;
  std::array<int64_t, 1> grid_stride;
  std::array<float, 2> coors_range;
  for (int i = 0; i < 1; ++i){
      vsize[0 - i] = vsize_xyz[i];
      coors_range[0 - i] = coors_range_xyz[i];
      coors_range[1 - i] = coors_range_xyz[i + 1];
  }
  int64_t prod = 1;
  for (size_t i = 0; i < 1; ++i) {
      grid_size[i] =
          std::round((coors_range[1 + i] - coors_range[i]) / vsize[i]);
  }
  for (int i = 1 - 1; i >= 0; --i) {
      grid_stride[i] = prod;
      prod *= grid_size[i];
  }
  return std::make_tuple(vsize, grid_size, grid_stride, coors_range);
}
} // namespace p2v_c
} // namespace ops_cpu1d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib