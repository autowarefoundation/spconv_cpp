#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTCKernel.h>
#include <spconvlib/cumm/common/GemmBasic.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f16tnt_m64n64k64m32n32k64A1T16832_200_C301LLL_SK_S8 {
namespace mma {
namespace mma_ns_wmma {
namespace tensorop {
using TensorViewNVRTCKernel = spconvlib::cumm::common::TensorViewNVRTCKernel;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
struct MmaSync {
  __forceinline__ __device__ void operator()(tv::array<int32_t, 4, 0>& d, tv::array<int8_t, 16, 0> const & a, tv::array<int8_t, 8, 0> const & b, tv::array<int32_t, 4, 0> const & c)   {
    
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
    #endif
  }
};
} // namespace tensorop
} // namespace mma_ns_wmma
} // namespace mma
} // namespace Ampere_s8s8s8s32f16tnt_m64n64k64m32n32k64A1T16832_200_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib