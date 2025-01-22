#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T884_200_S1 {
namespace mma {
namespace mma_ns_sb {
namespace layout {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct VoltaTensorOpCrosswise {
  int32_t stride;
  __forceinline__ __host__ __device__ constexpr  VoltaTensorOpCrosswise(int32_t stride_) : stride(stride_)  {
    
  }
  __forceinline__ __host__ __device__ constexpr static VoltaTensorOpCrosswise from_shape(const tv::array<int, 2> & shape)   {
    return VoltaTensorOpCrosswise(shape[1]);
  }
  __forceinline__ __host__ __device__ int64_t operator()(int32_t x, int32_t y)  const {
    int vec_contiguous_idx = y / 4;
    int vec_strided_idx = x;
    int vec_strided_within_tile = vec_contiguous_idx & 0x7;
    // 0: tile: 4x64, a smem bank
    // 1. map to tile offset. assume we have 4x128, so tile offset
    // is 0, 64, 128, 192, ...
    int permuted_vec_contiguous  =  vec_strided_idx & (~0xF);
    // 2. inside a tile, map to each permuted sub tile 4x16
    // (0,4,8,12)[], (0,16,32,48)[]
    permuted_vec_contiguous += (vec_strided_idx & 0x3) * 4;
    permuted_vec_contiguous += (((vec_strided_idx >> 2) ^ ((vec_strided_idx & 0x10) >> 3)) & 0x3);
    // 3. generate permuted offset
    permuted_vec_contiguous ^= ((vec_strided_within_tile >> 1) & 0x3);
    int permuted_vec_strided = vec_contiguous_idx;
    int element_contiguous = permuted_vec_contiguous *  4 + 
                            (y % 4);
    return element_contiguous + permuted_vec_strided * (stride * 4);
  }
};
} // namespace layout
} // namespace mma_ns_sb
} // namespace mma
} // namespace Volta_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib