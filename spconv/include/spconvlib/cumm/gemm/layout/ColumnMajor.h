#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace layout {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct ColumnMajor {
  int32_t stride;
  __forceinline__ __host__ __device__ constexpr  ColumnMajor(int32_t stride_) : stride(stride_)  {
    
  }
  __forceinline__ __host__ __device__ constexpr static ColumnMajor from_shape(const tv::array<int, 2> & shape)   {
    return ColumnMajor(shape[0]);
  }
  __forceinline__ __host__ __device__ constexpr int64_t operator()(int32_t x, int32_t y)  const {
    return int64_t(y) * int64_t(stride) + x;
  }
  __forceinline__ __host__ __device__ constexpr int32_t inverse_0(int64_t offset)  const {
    return int32_t(offset % stride);
  }
  __forceinline__ __host__ __device__ constexpr int32_t inverse_1(int64_t offset)  const {
    return int32_t(offset / stride);
  }
};
} // namespace layout
} // namespace gemm
} // namespace cumm
} // namespace spconvlib