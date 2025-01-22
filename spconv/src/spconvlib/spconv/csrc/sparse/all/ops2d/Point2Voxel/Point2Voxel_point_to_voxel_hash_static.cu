#include <spconvlib/spconv/csrc/sparse/all/ops2d/Point2Voxel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops2d/kernel/Point2VoxelKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops2d {
using TensorView = spconvlib::cumm::common::TensorView;
using Point2VoxelCommon = spconvlib::spconv::csrc::sparse::all::ops2d::p2v_c::Point2VoxelCommon;
using Layout = spconvlib::spconv::csrc::sparse::all::ops2d::layout_ns::TensorGeneric;
std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> Point2Voxel::point_to_voxel_hash_static(tv::Tensor points, tv::Tensor voxels, tv::Tensor indices, tv::Tensor num_per_voxel, tv::Tensor hashdata, tv::Tensor point_indice_data, tv::Tensor points_voxel_id, std::array<float, 2> vsize, std::array<int, 2> grid_size, std::array<int64_t, 2> grid_stride, std::array<float, 4> coors_range, bool clear_voxels, bool empty_mean, std::uintptr_t stream_int)   {
  
  auto vsize_tv = Point2VoxelCommon::array2tvarray(vsize);
  auto grid_size_tv = Point2VoxelCommon::array2tvarray(grid_size);
  auto grid_stride_tv = Point2VoxelCommon::array2tvarray(grid_stride);
  auto coors_range_tv = Point2VoxelCommon::array2tvarray(coors_range);
  auto custream = reinterpret_cast<cudaStream_t>(stream_int);
  auto ctx = tv::Context();
  ctx.set_cuda_stream(custream);
  TV_ASSERT_INVALID_ARG(points.ndim() == 2 && points.dim(1) >= 2, "error");
  using V = int64_t;
  using KeyType = int64_t;
  constexpr KeyType kEmptyKey = std::numeric_limits<KeyType>::max();
  if (clear_voxels){
      voxels.zero_(ctx);
  }
  using table_t =
      tv::hash::LinearHashTable<KeyType, V, tv::hash::Murmur3Hash<KeyType>,
                                  kEmptyKey, false>;
  using pair_t = typename table_t::value_type;
  // int64_t expected_hash_data_num = int64_t(tv::hash::align_to_power2(points.dim(0) * 2));
  int64_t expected_hash_data_num = points.dim(0) * 2;
  TV_ASSERT_RT_ERR(hashdata.dim(0) >= expected_hash_data_num, "hash table too small")
  TV_ASSERT_RT_ERR(point_indice_data.dim(0) >= points.dim(0), "point_indice_data too small")
  num_per_voxel.zero_(ctx);
  table_t hash = table_t(hashdata.data_ptr<pair_t>(), expected_hash_data_num);
  tv::hash::clear_map(hash, custream);
  auto launcher = tv::cuda::Launch(points.dim(0), custream);
  launcher(kernel::build_hash_table<table_t>, hash, points.data_ptr<const float>(),
          point_indice_data.data_ptr<int64_t>(),
          points.dim(1), vsize_tv, coors_range_tv, grid_size_tv, grid_stride_tv, points.dim(0));
  auto table_launcher = tv::cuda::Launch(hash.size(), custream);
  tv::Tensor count = tv::zeros({1}, tv::int32, 0);
  Layout layout = Layout::from_shape(grid_size_tv);
  table_launcher(kernel::assign_table<table_t>, hash, indices.data_ptr<int>(),
                  count.data_ptr<int>(),
                  layout, voxels.dim(0));
  launcher(kernel::generate_voxel<table_t>, hash, points.data_ptr<const float>(),
          point_indice_data.data_ptr<const int64_t>(), voxels.data_ptr<float>(),
          num_per_voxel.data_ptr<int>(), points_voxel_id.data_ptr<int64_t>(), points.dim(1), voxels.dim(1), 
          voxels.dim(0), vsize_tv, coors_range_tv,
          grid_size_tv, grid_stride_tv, points.dim(0));
  auto count_cpu = count.cpu();
  int count_val = count_cpu.item<int32_t>();
  count_val = count_val > voxels.dim(0) ? voxels.dim(0) : count_val;
  auto voxel_launcher = tv::cuda::Launch(count_val, custream);
  if (empty_mean){
      launcher(kernel::voxel_empty_fill_mean, voxels.data_ptr<float>(),
              num_per_voxel.data_ptr<int>(), count_val, 
              voxels.dim(1), voxels.dim(2));
  }else{
      launcher(kernel::limit_num_per_voxel_value, num_per_voxel.data_ptr<int>(), count_val, 
              voxels.dim(1));
  }
  return std::make_tuple(voxels.slice_first_axis(0, count_val), 
      indices.slice_first_axis(0, count_val), 
      num_per_voxel.slice_first_axis(0, count_val));
}
} // namespace ops2d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib