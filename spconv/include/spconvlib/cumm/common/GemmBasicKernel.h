#pragma once
#include <tensorview/gemm/arch/memory.h>
#include <tensorview/gemm/arch/transpose.h>
#include <tensorview/gemm/arch/semaphore.h>
#include <spconvlib/cumm/common/GemmBasic.h>
namespace spconvlib {
namespace cumm {
namespace common {
using GemmBasic = spconvlib::cumm::common::GemmBasic;
struct GemmBasicKernel {
};
} // namespace common
} // namespace cumm
} // namespace spconvlib