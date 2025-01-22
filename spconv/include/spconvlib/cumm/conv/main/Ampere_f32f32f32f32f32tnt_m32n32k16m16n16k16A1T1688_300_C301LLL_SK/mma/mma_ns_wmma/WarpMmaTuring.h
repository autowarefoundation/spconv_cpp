#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/conv/main/Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK/mma/mma_ns_wmma/tensorop/MmaSync.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK {
namespace mma {
namespace mma_ns_wmma {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using InstMma = spconvlib::cumm::conv::main::Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK::mma::mma_ns_wmma::tensorop::MmaSync;
struct WarpMmaTuring {
  __forceinline__ __device__ void operator()(tv::array<float, 8, 0>& D, tv::array<float, 4, 0> const & A, tv::array<float, 4, 0> const & B, tv::array<float, 8, 0> const & C)   {
    
    InstMma mma;
    D = C;
    tv::array<tv::tfloat32_t, 4, 0> const *ptr_A = reinterpret_cast<tv::array<tv::tfloat32_t, 4, 0> const *>(&A);
    tv::array<tv::tfloat32_t, 2, 0> const *ptr_B = reinterpret_cast<tv::array<tv::tfloat32_t, 2, 0> const *>(&B);
    tv::array<float, 4, 0> *ptr_D = reinterpret_cast<tv::array<float, 4, 0> *>(&D);
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
      TV_PRAGMA_UNROLL
      for (int n = 0; n < 2; ++n){
        TV_PRAGMA_UNROLL
        for (int m = 0; m < 1; ++m){
          int m_serpentine = ((n % 2) ? (1 - 1 - m) : m);
          mma(ptr_D[m_serpentine + n * 1],
              ptr_A[m_serpentine],
              ptr_B[n],
              ptr_D[m_serpentine + n * 1]);
        }
      }
    #elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
      TV_PRAGMA_UNROLL
      for (int m = 0; m < 1; ++m){
        TV_PRAGMA_UNROLL
        for (int n = 0; n < 2; ++n){
          int n_serpentine = ((m % 2) ? (2 - 1 - n) : n);
          mma(ptr_D[m + n_serpentine * 1],
              ptr_A[m],
              ptr_B[n_serpentine],
              ptr_D[m + n_serpentine * 1]);
        }
      }
    #endif
  }
};
} // namespace mma_ns_wmma
} // namespace mma
} // namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib