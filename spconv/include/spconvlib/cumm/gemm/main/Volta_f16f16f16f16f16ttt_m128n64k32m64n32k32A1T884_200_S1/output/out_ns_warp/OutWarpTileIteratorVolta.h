#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16ttt_m128n64k32m64n32k32A1T884_200_S1 {
namespace output {
namespace out_ns_warp {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
struct OutWarpTileIteratorVolta {
  tv::alignedarray<int2, 1, 8> * pointer_;
  __forceinline__ __device__  OutWarpTileIteratorVolta(tv::half_t * ptr, int warp_offset_m, int warp_offset_n, int lane_idx)   {
    
    int quad_id = lane_idx / 4; // 0000 1111 2222 3333 ....
    int lane_in_quad = (lane_idx % 4);
    int quad_row_idx = ((quad_id & 4) >> 1) + (quad_id & 1); //
    int quad_col_idx = ((quad_id & 2) >> 1);
    int row = quad_row_idx * 4 + lane_in_quad;
    int column = quad_col_idx * 8 / 4;
    pointer_ = reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(ptr) + row * 17 +
                column;
    add_warp_offset(warp_offset_m, warp_offset_n);
  }
  __forceinline__ __device__ void add_warp_offset(int warp_m, int warp_n)   {
    pointer_ +=
        (warp_m * 16 * (64 + 4) + warp_n * 32) /
        4;
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<tv::half_t, 16, 0> const & frag, int32_t pointer_offset)   {
    const tv::alignedarray<int2, 1, 8> * frag_ptr = reinterpret_cast<const tv::alignedarray<int2, 1, 8> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int tile_idx = 0; tile_idx < 1; ++tile_idx) {
        TV_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < 4; ++access_idx) {
            int access_quad = access_idx / 2;
            int access = access_idx % 2;
            int ptr_offset =
                tile_idx * 32 / 4 +
                access_quad * 16 / 4 + access +
                pointer_offset / 4;
            int frag_idx = tile_idx * 4 + access_idx;
            pointer_[ptr_offset] = frag_ptr[frag_idx];
        }
    }
  }
  __forceinline__ __device__ void store(tv::array<tv::half_t, 16, 0> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    pointer_ += pointer_offset / 4;
  }
};
} // namespace out_ns_warp
} // namespace output
} // namespace Volta_f16f16f16f16f16ttt_m128n64k32m64n32k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib