#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK {
namespace out_smem_storage {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct OutputSmemStorage {
  tv::alignedarray<float, 640, 16> smem;
};
} // namespace out_smem_storage
} // namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib