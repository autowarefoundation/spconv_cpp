#pragma once
#include <spconvlib/cumm/conv/params/ConvProblemCommon.h>
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace cpAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK {
namespace cp {
using ConvProblemCommon = spconvlib::cumm::conv::params::ConvProblemCommon;
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct ConvProblem {
  int N;
  int C;
  int K;
  int kernel_volume;
  tv::gemm::ConvMode mode;
  int split_k_slices;
  int groups;
  TV_HOST_DEVICE_INLINE  ConvProblem()   {
    
  }
  TV_HOST_DEVICE_INLINE  ConvProblem(int N, int C, int K, int kernel_volume, tv::gemm::ConvMode mode = tv::gemm::ConvMode::kCrossCorrelation, int split_k_slices = 1, int groups = 1) : N(N), C(C), K(K), kernel_volume(kernel_volume), mode(mode), split_k_slices(split_k_slices), groups(groups)  {
    
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 4> get_npq_shape()   {
    
    return {N};
  }
  TV_HOST_DEVICE_INLINE bool check_npq_not_overflow()   {
    
    auto shape = get_npq_shape();
    return (
    std::abs(int64_t(shape[0]) * int64_t(shape[1]) * int64_t(shape[2]) * int64_t(shape[3])) <= std::numeric_limits<int>::max()
    );
  }
  TV_HOST_DEVICE_INLINE static tv::array<int, 3> calc_output_dims(tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> padding, tv::array<int, 3> stride, tv::array<int, 3> dilation)   {
    
    tv::array<int, 3> out;
    for (int i = 0; i < 3; ++i){
        out[i] = ((input_dims[i] + padding[i] * 2 - ksize[i] * dilation[i]) / stride[i]) + 1;
    }
    return out;
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 3> implicit_gemm_mnk(tv::gemm::ConvOpType op_type)   {
    
    return ConvProblemCommon::implicit_gemm_mnk(op_type, N, C, K, kernel_volume, -1, -1, true);
  }
  TV_HOST_DEVICE_INLINE int implicit_gemm_k_iterations(tv::gemm::ConvOpType op_type, int tile_shape_k)   {
    
    switch (op_type) {
        case tv::gemm::ConvOpType::kForward:
            return kernel_volume * tv::div_up(tv::div_up(C, split_k_slices), tile_shape_k);
        case tv::gemm::ConvOpType::kBackwardInput:
            return kernel_volume * tv::div_up(tv::div_up(K, split_k_slices), tile_shape_k);
        case tv::gemm::ConvOpType::kBackwardWeight:
            return tv::div_up(tv::div_up(N, split_k_slices), tile_shape_k);
        default:
            return 0;
    }
    return 0;
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 2> get_input_shape()   {
    
    return {N, C};
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 3> get_weight_shape()   {
    
    return {K, kernel_volume, C};
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 2> get_output_shape()   {
    
    return {N, K};
  }
};
} // namespace cp
} // namespace cpAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib