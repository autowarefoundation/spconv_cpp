#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTCKernel.h>
#include <spconvlib/cumm/common/GemmBasic.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK {
namespace mma {
namespace mma_ns_wmma {
namespace tensorop {
using TensorViewNVRTCKernel = spconvlib::cumm::common::TensorViewNVRTCKernel;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
struct MmaSync {
  __forceinline__ __device__ void operator()(tv::array<float, 4, 0>& d, tv::array<tv::tfloat32_t, 4, 0> const & a, tv::array<tv::tfloat32_t, 2, 0> const & b, tv::array<float, 4, 0> const & c)   {
    
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
    #endif
  }
};
} // namespace tensorop
} // namespace mma_ns_wmma
} // namespace mma
} // namespace Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib