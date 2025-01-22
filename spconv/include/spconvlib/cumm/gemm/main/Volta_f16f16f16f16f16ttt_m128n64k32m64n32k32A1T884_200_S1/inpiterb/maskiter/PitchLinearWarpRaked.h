#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16ttt_m128n64k32m64n32k32A1T884_200_S1 {
namespace inpiterb {
namespace maskiter {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct PitchLinearWarpRaked {
  __forceinline__ __host__ __device__ static bool is_skipped(int thread_id)   {
    
    if(!0)
        return false;
    return thread_id >= 128;
  }
  __forceinline__ __host__ __device__ static tv::array<int, 2> initial_offset(int thread_id)   {
    if(is_skipped(thread_id))
        return {0, 0};        // to InputIter: inefficient but convenience dummy offset.
    int warp_id = (thread_id / 32);
    int lane_id = (thread_id % 32);
    tv::array<int, 2> warp_offset{warp_id / 1,
                                warp_id % 1};
    constexpr tv::array<int, 2> kWarpDilation{8, 8};
    tv::array<int, 2> thread_offset_in_warp{lane_id / 8,
                                            lane_id % 8};
    tv::array<int, 2> offset_in_tile =
        kWarpDilation * warp_offset + thread_offset_in_warp;
    return {offset_in_tile[0], offset_in_tile[1] * 8};
  }
};
} // namespace maskiter
} // namespace inpiterb
} // namespace Volta_f16f16f16f16f16ttt_m128n64k32m64n32k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib