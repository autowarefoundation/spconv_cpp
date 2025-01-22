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
std::tuple<std::array<float, 1>, std::array<int, 1>, std::array<int64_t, 1>, std::array<float, 2>> Point2VoxelCPU::calc_meta_data(std::array<float, 1> vsize_xyz, std::array<float, 2> coors_range_xyz)   {
  
  return Point2VoxelCommon::calc_meta_data(vsize_xyz, coors_range_xyz);
}
} // namespace ops_cpu1d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib