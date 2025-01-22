#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m64n256k8m32n64k8A1_200_S1 {
namespace inpiterb {
namespace maskiter {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct PitchLinear {
  __forceinline__ __host__ __device__ static tv::array<int, 2> initial_offset(int thread_id)   {
    return {(thread_id / 8) * 1,
            (thread_id % 8) *  1};
  }
};
} // namespace maskiter
} // namespace inpiterb
} // namespace Simt_f32f32f32f32f32tnt_m64n256k8m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib