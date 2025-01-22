#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_C301LLL_SK {
namespace inpiterb {
namespace tmap {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct PitchLinear {
  __forceinline__ __host__ __device__ static tv::array<int, 2> initial_offset(int thread_id)   {
    return {(thread_id / 32) * 1,
            (thread_id % 32) *  1};
  }
};
} // namespace tmap
} // namespace inpiterb
} // namespace Simt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib