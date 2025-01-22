#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK {
namespace inpitera {
namespace tmap {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct PitchLinear {
  __forceinline__ __host__ __device__ static tv::array<int, 2> initial_offset(int thread_id)   {
    return {(thread_id / 16) * 1,
            (thread_id % 16) *  1};
  }
};
} // namespace tmap
} // namespace inpitera
} // namespace Simt_f32f32f32f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib