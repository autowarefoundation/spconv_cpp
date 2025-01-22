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
std::tuple<std::array<float, 4>, std::array<int, 4>, std::array<int64_t, 4>, std::array<float, 8>> Point2VoxelCPU::calc_meta_data(std::array<float, 4> vsize_xyz, std::array<float, 8> coors_range_xyz)   {
  
  return Point2VoxelCommon::calc_meta_data(vsize_xyz, coors_range_xyz);
}
} // namespace ops_cpu4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib