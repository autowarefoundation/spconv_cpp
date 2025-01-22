#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTCKernel.h>
#include <spconvlib/cumm/common/GemmBasic.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Volta_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T884_200_C301LLL_SK {
namespace mma {
namespace mma_ns_wmma {
namespace tensorop {
using TensorViewNVRTCKernel = spconvlib::cumm::common::TensorViewNVRTCKernel;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
struct MmaSync {
  __forceinline__ __device__ void operator()(tv::array<float, 8, 0>& d, tv::array<tv::half_t, 4, 0> const & a, tv::array<tv::half_t, 4, 0> const & b, tv::array<float, 8, 0> const & c)   {
    
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    #endif
  }
};
} // namespace tensorop
} // namespace mma_ns_wmma
} // namespace mma
} // namespace Volta_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T884_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib