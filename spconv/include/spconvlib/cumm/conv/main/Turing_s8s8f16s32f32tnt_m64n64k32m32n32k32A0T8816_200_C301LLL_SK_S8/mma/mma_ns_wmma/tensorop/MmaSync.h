#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTCKernel.h>
#include <spconvlib/cumm/common/GemmBasic.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8f16s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8 {
namespace mma {
namespace mma_ns_wmma {
namespace tensorop {
using TensorViewNVRTCKernel = spconvlib::cumm::common::TensorViewNVRTCKernel;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
struct MmaSync {
  __forceinline__ __device__ void operator()(tv::array<int32_t, 2, 0>& d, tv::array<int8_t, 4, 0> const & a, tv::array<int8_t, 4, 0> const & b, tv::array<int32_t, 2, 0> const & c)   {
    
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
    unsigned const & A = reinterpret_cast<unsigned const &>(a);
    unsigned const & B = reinterpret_cast<unsigned const &>(b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
    #endif
  }
};
} // namespace tensorop
} // namespace mma_ns_wmma
} // namespace mma
} // namespace Turing_s8s8f16s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib