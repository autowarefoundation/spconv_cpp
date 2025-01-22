#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8 {
namespace out_smem_storage {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct OutputSmemStorage {
  tv::alignedarray<int32_t, 1152, 16> smem;
};
} // namespace out_smem_storage
} // namespace Ampere_s8s8f32s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib