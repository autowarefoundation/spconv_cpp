#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Volta_f16f16f16f32f32tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK {
namespace output {
namespace out_ns_warp {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
struct OutWarpTileIteratorVolta {
  tv::alignedarray<int2, 1, 8> * pointer_;
  __forceinline__ __device__  OutWarpTileIteratorVolta(float * ptr, int warp_offset_m, int warp_offset_n, int lane_idx)   {
    
    int quad_id = lane_idx / 4;
    int lane_in_quad = (lane_idx % 4);
    int const kQuadRowDelta = 4;
    int const kQuadColumnDelta = 2 * 2;
    int quad_row_offset = ((quad_id & 4) / 2 + (quad_id & 1)) * kQuadRowDelta;
    int quad_column_offset = (quad_id & 2) / 2 * kQuadColumnDelta;
    int thread_row_offset = (lane_in_quad & 1);
    int thread_column_offset = (lane_in_quad & 2) / 2;
    int row = quad_row_offset + thread_row_offset;
    int column = quad_column_offset + thread_column_offset;
    pointer_ = reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(ptr) + row * 129 +
                column;
    add_warp_offset(warp_offset_m, warp_offset_n);
  }
  __forceinline__ __device__ void add_warp_offset(int warp_m, int warp_n)   {
    pointer_ +=
        (warp_m * 16 * (256 + 2) + warp_n * 64) /
        2;
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<float, 32, 0> const & frag, int32_t pointer_offset)   {
    const tv::alignedarray<int2, 1, 8> * frag_ptr = reinterpret_cast<const tv::alignedarray<int2, 1, 8> *>(&frag);
    int const kAccessesPerRow = 2 * 2  * 2;
    TV_PRAGMA_UNROLL
    for (int row_idx = 0; row_idx < 2; ++row_idx) {
        TV_PRAGMA_UNROLL
        for (int access_idx = 0; access_idx < kAccessesPerRow; ++access_idx) {
            int frag_idx = row_idx * kAccessesPerRow + access_idx;
            int ptr_column_offset = (access_idx & 1) * 2 +
                                    (access_idx & 2) * 2  * 2 +
                                    (access_idx & 4) * 2  * 2;
            int ptr_row_offset = row_idx * 2;
            int ptr_offset = ptr_row_offset * 129 +
                            ptr_column_offset +
                            pointer_offset / 2;
            pointer_[ptr_offset] = frag_ptr[frag_idx];
        }
    }
  }
  __forceinline__ __device__ void store(tv::array<float, 32, 0> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    pointer_ += pointer_offset / 2;
  }
};
} // namespace out_ns_warp
} // namespace output
} // namespace Volta_f16f16f16f32f32tnt_m32n256k32m32n64k32A1T884_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib