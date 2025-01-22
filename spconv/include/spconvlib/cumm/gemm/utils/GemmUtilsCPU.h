#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace utils {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct GemmUtilsCPU {
  TV_HOST_DEVICE_INLINE static tv::array<int, 3> get_logical_tile_count(int m, int n, int k, int tile_m, int tile_n, int split_k_slices)   {
    
    tv::array<int, 3> grid_dims;
    grid_dims[0] = tv::div_up(m, tile_m);
    grid_dims[1] = tv::div_up(n, tile_n);
    grid_dims[2] = split_k_slices;
    return grid_dims;
  }
  TV_HOST_DEVICE_INLINE static int get_gemm_k_size_per_split(int k, int split_k, int tile_k)   {
    
    int total_gemm_k_iterations = tv::div_up(k, tile_k);
    int gemm_k_iterations_per_split =
        tv::div_up(total_gemm_k_iterations, split_k);
    auto gemm_k_size_per_split = gemm_k_iterations_per_split * tile_k; 
    return gemm_k_size_per_split;
  }
};
} // namespace utils
} // namespace gemm
} // namespace cumm
} // namespace spconvlib