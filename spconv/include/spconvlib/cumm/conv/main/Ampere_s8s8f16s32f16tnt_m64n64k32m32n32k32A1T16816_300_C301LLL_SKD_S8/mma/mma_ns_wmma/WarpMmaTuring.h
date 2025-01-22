#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8f16s32f16tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8/mma/mma_ns_wmma/tensorop/MmaSync.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8f16s32f16tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8 {
namespace mma {
namespace mma_ns_wmma {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using InstMma = spconvlib::cumm::conv::main::Ampere_s8s8f16s32f16tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8::mma::mma_ns_wmma::tensorop::MmaSync;
struct WarpMmaTuring {
  __forceinline__ __device__ void operator()(tv::array<int32_t, 32, 0>& D, tv::array<int8_t, 16, 0> const & A, tv::array<int8_t, 16, 0> const & B, tv::array<int32_t, 32, 0> const & C)   {
    
    InstMma mma;
    D = C;
    tv::array<int8_t, 8, 0> const *ptr_A = reinterpret_cast<tv::array<int8_t, 8, 0> const *>(&A);
    tv::array<int8_t, 4, 0> const *ptr_B = reinterpret_cast<tv::array<int8_t, 4, 0> const *>(&B);
    tv::array<int32_t, 4, 0> *ptr_D = reinterpret_cast<tv::array<int32_t, 4, 0> *>(&D);
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
      TV_PRAGMA_UNROLL
      for (int n = 0; n < 4; ++n){
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
        for (int n = 0; n < 4; ++n){
          int n_serpentine = ((m % 2) ? (4 - 1 - n) : n);
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
} // namespace Ampere_s8s8f16s32f16tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib