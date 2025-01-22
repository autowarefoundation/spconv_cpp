#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32ttt_m32n64k16m32n32k8A1_200_S1/mma/mma_ns_wa/ns2/RowMajorInterleaved.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32ttt_m32n64k16m32n32k8A1_200_S1 {
namespace output {
namespace out_ns_warp {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using LaneLayout = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32ttt_m32n64k16m32n32k8A1_200_S1::mma::mma_ns_wa::ns2::RowMajorInterleaved;
struct OutWarpTileIterator {
  tv::alignedarray<int, 1, 4> * pointer_;
  __forceinline__ __device__  OutWarpTileIterator(float * ptr, int warp_offset_m, int warp_offset_n, int lane_idx)   {
    constexpr auto lane_layout = LaneLayout::from_shape({4, 8});
    tv::array<int, 2> logical_offset{warp_offset_m, warp_offset_n};
    // 0, 1, 0, 1, 0, 1, ..., 2, 3, 2, 3, 2, 3
    int lane_offset_0 = lane_layout.inverse_0(lane_idx);
    // 0, 0, 1, 1, 2, 2, 3, 3, ...
    int lane_offset_1 = lane_layout.inverse_1(lane_idx);
    // saved to compacted shared memory, so logical_offset[0] * warp_shape[0],
    // not logical_offset[0] * warp_tile_shape[0]
    pointer_ = reinterpret_cast<tv::alignedarray<int, 1, 4> *>(
        ptr + (logical_offset[0] * 4 + lane_offset_0) * 81 +
        logical_offset[1] * 32 +
        lane_offset_1 * 4);
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<float, 4, 0> const& frag, int32_t pointer_offset)   {
    // pointer_offset: element unit
    const tv::alignedarray<int, 1, 4> * dst_ptr =
        reinterpret_cast<const tv::alignedarray<int, 1, 4> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int acc_idx = 0; acc_idx < 1; ++acc_idx) {
        if (true) {
            TV_PRAGMA_UNROLL
            for (int s = 0; s < 4; ++s) {
                pointer_[acc_idx * 8 * 4 + s +
                        pointer_offset] = dst_ptr[acc_idx * 4 + s];
            }
        } else {
            pointer_[acc_idx * 8 +
                    pointer_offset / 4] = dst_ptr[acc_idx];
        }
    }
  }
  __forceinline__ __device__ void store(tv::array<float, 4, 0> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    pointer_ += pointer_offset / 1;
  }
};
} // namespace out_ns_warp
} // namespace output
} // namespace Simt_f32f32f32f32f32ttt_m32n64k16m32n32k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib