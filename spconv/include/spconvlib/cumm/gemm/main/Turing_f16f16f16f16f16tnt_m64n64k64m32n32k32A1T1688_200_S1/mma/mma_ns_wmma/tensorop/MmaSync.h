#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTCKernel.h>
#include <spconvlib/cumm/common/GemmBasic.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1 {
namespace mma {
namespace mma_ns_wmma {
namespace tensorop {
using TensorViewNVRTCKernel = spconvlib::cumm::common::TensorViewNVRTCKernel;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
struct MmaSync {
  __forceinline__ __device__ void operator()(tv::array<tv::half_t, 4, 0>& d, tv::array<tv::half_t, 4, 0> const & a, tv::array<tv::half_t, 2, 0> const & b, tv::array<tv::half_t, 4, 0> const & c)   {
    
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const & B = reinterpret_cast<unsigned const &>(b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]));
    #endif
  }
};
} // namespace tensorop
} // namespace mma_ns_wmma
} // namespace mma
} // namespace Turing_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib