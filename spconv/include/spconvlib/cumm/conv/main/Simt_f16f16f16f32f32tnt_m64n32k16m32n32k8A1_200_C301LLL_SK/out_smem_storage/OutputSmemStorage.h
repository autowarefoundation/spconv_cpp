#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f16f16f16f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK {
namespace out_smem_storage {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct OutputSmemStorage {
  tv::alignedarray<float, 784, 16> smem;
};
} // namespace out_smem_storage
} // namespace Simt_f16f16f16f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib