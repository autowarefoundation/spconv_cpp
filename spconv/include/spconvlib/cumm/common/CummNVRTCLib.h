#pragma once
#include <tensorview/cuda/nvrtc.h>
#include <tensorview/gemm/core/nvrtc_bases.h>
#include <spconvlib/cumm/common/TensorView.h>
namespace spconvlib {
namespace cumm {
namespace common {
using TensorView = spconvlib::cumm::common::TensorView;
struct CummNVRTCLib {
};
} // namespace common
} // namespace cumm
} // namespace spconvlib