#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/gemm/main/Turing_f16f16f16f16f16ttt_m64n128k32m32n64k32A1T1688_200_S1/mma/mma_ns_wmma/tensorop/MmaSync.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_f16f16f16f16f16ttt_m64n128k32m32n64k32A1T1688_200_S1 {
namespace mma {
namespace mma_ns_wmma {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using InstMma = spconvlib::cumm::gemm::main::Turing_f16f16f16f16f16ttt_m64n128k32m32n64k32A1T1688_200_S1::mma::mma_ns_wmma::tensorop::MmaSync;
struct WarpMmaTuring {
  __forceinline__ __device__ void operator()(tv::array<tv::half_t, 64, 0>& D, tv::array<tv::half_t, 8, 0> const & A, tv::array<tv::half_t, 16, 0> const & B, tv::array<tv::half_t, 64, 0> const & C)   {
    
    InstMma mma;
    D = C;
    tv::array<tv::half_t, 4, 0> const *ptr_A = reinterpret_cast<tv::array<tv::half_t, 4, 0> const *>(&A);
    tv::array<tv::half_t, 2, 0> const *ptr_B = reinterpret_cast<tv::array<tv::half_t, 2, 0> const *>(&B);
    tv::array<tv::half_t, 4, 0> *ptr_D = reinterpret_cast<tv::array<tv::half_t, 4, 0> *>(&D);
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
      TV_PRAGMA_UNROLL
      for (int n = 0; n < 8; ++n){
        TV_PRAGMA_UNROLL
        for (int m = 0; m < 2; ++m){
          int m_serpentine = ((n % 2) ? (2 - 1 - m) : m);
          mma(ptr_D[m_serpentine + n * 2],
              ptr_A[m_serpentine],
              ptr_B[n],
              ptr_D[m_serpentine + n * 2]);
        }
      }
    #elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
      TV_PRAGMA_UNROLL
      for (int m = 0; m < 2; ++m){
        TV_PRAGMA_UNROLL
        for (int n = 0; n < 8; ++n){
          int n_serpentine = ((m % 2) ? (8 - 1 - n) : n);
          mma(ptr_D[m + n_serpentine * 2],
              ptr_A[m],
              ptr_B[n_serpentine],
              ptr_D[m + n_serpentine * 2]);
        }
      }
    #endif
  }
};
} // namespace mma_ns_wmma
} // namespace mma
} // namespace Turing_f16f16f16f16f16ttt_m64n128k32m32n64k32A1T1688_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib