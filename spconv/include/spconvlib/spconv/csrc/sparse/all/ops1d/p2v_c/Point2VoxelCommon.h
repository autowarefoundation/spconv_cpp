#pragma once
#include <spconvlib/cumm/common/TensorView.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops1d {
namespace p2v_c {
using TensorView = spconvlib::cumm::common::TensorView;
struct Point2VoxelCommon {
  /**
   * @param vsize_xyz 
   * @param coors_range_xyz 
   */
  static std::tuple<std::array<float, 1>, std::array<int, 1>, std::array<int64_t, 1>, std::array<float, 2>> calc_meta_data(std::array<float, 1> vsize_xyz, std::array<float, 2> coors_range_xyz);
  template <typename T, size_t N>
  static tv::array<T, N> array2tvarray(std::array<T, N> arr)   {
    
    tv::array<T, N> tarr;
    for (int i = 0; i < N; ++i){
        tarr[i] = arr[i];
    }
    return tarr;
  }
  template <typename T, size_t N>
  static std::array<T, N> tvarray2array(tv::array<T, N> arr)   {
    
    std::array<T, N> tarr;
    for (int i = 0; i < N; ++i){
        tarr[i] = arr[i];
    }
    return tarr;
  }
};
} // namespace p2v_c
} // namespace ops1d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib