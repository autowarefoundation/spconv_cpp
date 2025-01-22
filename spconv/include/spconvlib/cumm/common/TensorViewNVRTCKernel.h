#pragma once
#include <tensorview/cuda/device_ops.h>
#include <tensorview/gemm/debug.h>
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace common {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct TensorViewNVRTCKernel {
};
} // namespace common
} // namespace cumm
} // namespace spconvlib