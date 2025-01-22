#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasic.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace params {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
struct ConvProblemCommon {
  TV_HOST_DEVICE_INLINE static tv::array<int, 3> implicit_gemm_mnk(tv::gemm::ConvOpType op_type, int N, int C, int K, int kernel_volume, int in_prod, int out_prod, bool mask_sparse)   {
    
    if (mask_sparse){
        switch (op_type) {
            case tv::gemm::ConvOpType::kForward:
                return {N, K, C * kernel_volume};
            case tv::gemm::ConvOpType::kBackwardInput:
                return {N, C, K * kernel_volume};
            case tv::gemm::ConvOpType::kBackwardWeight:
                return {K, C * kernel_volume, N};
            default:
                return {};
        }
        return {};
    }else{
        switch (op_type) {
            case tv::gemm::ConvOpType::kForward:
                return {N * out_prod, K, C * kernel_volume};
            case tv::gemm::ConvOpType::kBackwardInput:
                return {N * in_prod, C, K * kernel_volume};
            case tv::gemm::ConvOpType::kBackwardWeight:
                return {K, C * kernel_volume, N * out_prod};
            default:
                return {};
        }
        return {};
    }
  }
  TV_HOST_DEVICE_INLINE static tv::array<int, 3> conv_iwo_012_to_abc(tv::gemm::ConvOpType op_type)   {
    
    switch (op_type) {
        case tv::gemm::ConvOpType::kForward:
            return {0, 1, 2};
        case tv::gemm::ConvOpType::kBackwardInput:
            return {2, 1, 0};
        case tv::gemm::ConvOpType::kBackwardWeight:
            return {1, 2, 0};
        default:
            return {};
    }
    return {};
  }
  TV_HOST_DEVICE_INLINE static tv::array<int, 3> gemm_abc_012_to_iwo(tv::gemm::ConvOpType op_type)   {
    
    switch (op_type) {
        case tv::gemm::ConvOpType::kForward:
            return {0, 1, 2};
        case tv::gemm::ConvOpType::kBackwardInput:
            return {2, 1, 0};
        case tv::gemm::ConvOpType::kBackwardWeight:
            return {2, 0, 1};
        default:
            return {};
    }
    return {};
  }
};
} // namespace params
} // namespace conv
} // namespace cumm
} // namespace spconvlib