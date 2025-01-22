#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1 {
namespace mma {
namespace mma_ns_wa {
namespace ns2 {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct RowMajorInterleaved {
  int32_t stride;
  __forceinline__ __host__ __device__ constexpr  RowMajorInterleaved(int32_t stride_) : stride(stride_)  {
    
  }
  __forceinline__ __host__ __device__ constexpr static RowMajorInterleaved from_shape(const tv::array<int, 2> & shape)   {
    return RowMajorInterleaved(shape[1] * 2);
  }
  __forceinline__ __host__ __device__ constexpr int64_t operator()(int32_t x, int32_t y)  const {
    int32_t row_major = x / 2;
    int32_t row_minor = x % 2;
    return int64_t(row_major) * int64_t(stride) +
        int64_t(y) * 2 + row_minor;
  }
  __forceinline__ __host__ __device__ constexpr int32_t inverse_0(int64_t offset)  const {
    int32_t row_major = int32_t(offset / stride);
    int32_t residual = int32_t(offset % stride);
    int32_t row_minor = residual % 2;
    return row_major * 2 + row_minor;
  }
  __forceinline__ __host__ __device__ constexpr int32_t inverse_1(int64_t offset)  const {
    return (offset % stride) / 2;
  }
};
} // namespace ns2
} // namespace mma_ns_wa
} // namespace mma
} // namespace Simt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib