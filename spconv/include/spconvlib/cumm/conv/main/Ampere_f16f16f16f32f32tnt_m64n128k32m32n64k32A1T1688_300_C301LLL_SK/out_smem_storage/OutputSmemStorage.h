#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_f16f16f16f32f32tnt_m64n128k32m32n64k32A1T1688_300_C301LLL_SK {
namespace out_smem_storage {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct OutputSmemStorage {
  tv::alignedarray<float, 4352, 16> smem;
};
} // namespace out_smem_storage
} // namespace Ampere_f16f16f16f32f32tnt_m64n128k32m32n64k32A1T1688_300_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib