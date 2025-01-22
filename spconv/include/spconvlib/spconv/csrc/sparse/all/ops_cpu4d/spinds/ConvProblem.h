#pragma once
#include <spconvlib/cumm/conv/params/ConvProblemCommon.h>
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu4d {
namespace spinds {
using ConvProblemCommon = spconvlib::cumm::conv::params::ConvProblemCommon;
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct ConvProblem {
  int N;
  int C;
  int K;
  tv::array<int, 4> input_dims;
  tv::array<int, 4> output_dims;
  tv::array<int, 4> ksize;
  tv::array<int, 4> padding;
  tv::array<int, 4> stride;
  tv::array<int, 4> dilation;
  tv::gemm::ConvMode mode;
  int split_k_slices;
  int groups;
  TV_HOST_DEVICE_INLINE  ConvProblem()   {
    
  }
  TV_HOST_DEVICE_INLINE  ConvProblem(int N, int C, int K, tv::array<int, 4> input_dims, tv::array<int, 4> output_dims, tv::array<int, 4> ksize, tv::array<int, 4> padding, tv::array<int, 4> stride, tv::array<int, 4> dilation, tv::gemm::ConvMode mode = tv::gemm::ConvMode::kCrossCorrelation, int split_k_slices = 1, int groups = 1) : N(N), C(C), K(K), input_dims(input_dims), output_dims(output_dims), ksize(ksize), padding(padding), stride(stride), dilation(dilation), mode(mode), split_k_slices(split_k_slices), groups(groups)  {
    
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 5> get_npq_shape()   {
    
    return {N, output_dims[0], output_dims[1], output_dims[2], output_dims[3]};
  }
  TV_HOST_DEVICE_INLINE bool check_npq_not_overflow()   {
    
    auto shape = get_npq_shape();
    return (
    std::abs(int64_t(shape[0]) * int64_t(shape[1]) * int64_t(shape[2]) * int64_t(shape[3]) * int64_t(shape[4])) <= std::numeric_limits<int>::max()
    );
  }
  TV_HOST_DEVICE_INLINE static tv::array<int, 4> calc_output_dims(tv::array<int, 4> input_dims, tv::array<int, 4> ksize, tv::array<int, 4> padding, tv::array<int, 4> stride, tv::array<int, 4> dilation)   {
    
    tv::array<int, 4> out;
    for (int i = 0; i < 4; ++i){
        out[i] = ((input_dims[i] + padding[i] * 2 - ksize[i] * dilation[i]) / stride[i]) + 1;
    }
    return out;
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 3> implicit_gemm_mnk(tv::gemm::ConvOpType op_type)   {
    
    int ksize_prod = ksize.op<tv::arrayops::prod>();
    int in_prod = input_dims.op<tv::arrayops::prod>();
    int out_prod = output_dims.op<tv::arrayops::prod>();
    return ConvProblemCommon::implicit_gemm_mnk(op_type, N, C, K, ksize_prod, in_prod, out_prod, false);
  }
  TV_HOST_DEVICE_INLINE int implicit_gemm_k_iterations(tv::gemm::ConvOpType op_type, int tile_shape_k)   {
    
    int ksize_prod = ksize.op<tv::arrayops::prod>();
    int in_prod = input_dims.op<tv::arrayops::prod>();
    int out_prod = output_dims.op<tv::arrayops::prod>();
    switch (op_type) {
        case tv::gemm::ConvOpType::kForward:
            return ksize_prod * tv::div_up(tv::div_up(C, split_k_slices), tile_shape_k);
        case tv::gemm::ConvOpType::kBackwardInput:
            return ksize_prod * tv::div_up(tv::div_up(K, split_k_slices), tile_shape_k);
        case tv::gemm::ConvOpType::kBackwardWeight:
            return tv::div_up(tv::div_up(N * out_prod, split_k_slices), tile_shape_k);
        default:
            return 0;
    }
    return 0;
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 6> get_input_shape()   {
    
    return {N, input_dims[0], input_dims[1], input_dims[2], input_dims[3], C};
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 6> get_weight_shape()   {
    
    return {K, ksize[0], ksize[1], ksize[2], ksize[3], C};
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 6> get_output_shape()   {
    
    return {N, output_dims[0], output_dims[1], output_dims[2], output_dims[3], K};
  }
};
} // namespace spinds
} // namespace ops_cpu4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib