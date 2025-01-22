#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m32n128k16m32n32k8A1_200_C301LLL_SK {
namespace out_smem_storage {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct OutputSmemStorage {
  tv::alignedarray<float, 1160, 16> smem;
};
} // namespace out_smem_storage
} // namespace Simt_f32f32f32f32f32tnt_m32n128k16m32n32k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib