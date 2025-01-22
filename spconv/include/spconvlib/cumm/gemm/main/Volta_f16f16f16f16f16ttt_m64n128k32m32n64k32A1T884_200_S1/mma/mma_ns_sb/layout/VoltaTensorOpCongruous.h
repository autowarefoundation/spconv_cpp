#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16ttt_m64n128k32m32n64k32A1T884_200_S1 {
namespace mma {
namespace mma_ns_sb {
namespace layout {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct VoltaTensorOpCongruous {
  int32_t stride;
  __forceinline__ __host__ __device__ constexpr  VoltaTensorOpCongruous(int32_t stride_) : stride(stride_)  {
    
  }
  __forceinline__ __host__ __device__ constexpr static VoltaTensorOpCongruous from_shape(const tv::array<int, 2> & shape)   {
    return VoltaTensorOpCongruous(shape[1]);
  }
  __forceinline__ __host__ __device__ int64_t operator()(int32_t x, int32_t y)  const {
    int vec_contiguous_idx = y / 8;
    int vec_strided_idx = x;
    // Compute the fundamental tile being accessed
    int tile_contiguous_idx = vec_contiguous_idx / 8;
    int tile_strided_idx = vec_strided_idx / 4;
    int tile_contiguous_residual = vec_contiguous_idx % 8;
    int tile_strided_residual = vec_strided_idx % 4;
    int permuted_strided_within_tile;
    int permuted_contiguous_within_tile;
    permuted_strided_within_tile = (tile_contiguous_residual & 0x3);
    permuted_contiguous_within_tile =
        (tile_strided_residual ^ permuted_strided_within_tile) |
        (tile_contiguous_residual & 0x4);
    // Compute final element location
    int element_contiguous = (tile_contiguous_idx * 8 +
                            permuted_contiguous_within_tile) *
                                8 +
                            (y % 8);
    int element_strided =
        tile_strided_idx * 4 + permuted_strided_within_tile;
    auto res = element_contiguous + element_strided * stride;
    // tv::printf2_block_once(threadIdx.x, stride_,
    // "VoltaTensorOpMultiplicandBCongruous", res, coord.strided(),
    // coord.contiguous());
    return res;
  }
};
} // namespace layout
} // namespace mma_ns_sb
} // namespace mma
} // namespace Volta_f16f16f16f16f16ttt_m64n128k32m32n64k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib