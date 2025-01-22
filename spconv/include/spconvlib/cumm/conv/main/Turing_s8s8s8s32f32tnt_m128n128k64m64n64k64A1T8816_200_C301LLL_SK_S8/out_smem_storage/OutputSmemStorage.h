#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8s8s32f32tnt_m128n128k64m64n64k64A1T8816_200_C301LLL_SK_S8 {
namespace out_smem_storage {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct OutputSmemStorage {
  tv::alignedarray<int32_t, 2176, 16> smem;
};
} // namespace out_smem_storage
} // namespace Turing_s8s8s8s32f32tnt_m128n128k64m64n64k64A1T8816_200_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib