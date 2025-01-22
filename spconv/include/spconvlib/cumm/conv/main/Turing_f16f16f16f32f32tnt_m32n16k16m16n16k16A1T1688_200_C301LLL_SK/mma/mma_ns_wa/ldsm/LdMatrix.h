#pragma once
#include <tensorview/gemm/arch/memory_sm75.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A1T1688_200_C301LLL_SK {
namespace mma {
namespace mma_ns_wa {
namespace ldsm {
struct LdMatrix {
  __forceinline__ __device__ static void run(tv::array<unsigned, 2, 0> & D, void const* ptr)   {
    
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
      unsigned addr = tv::gemm::get_smem_pointer(ptr);
      int x, y;
      asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(addr));
      reinterpret_cast<int2 &>(D) = make_int2(x, y);
    #else
      assert(0);
    #endif
  }
};
} // namespace ldsm
} // namespace mma_ns_wa
} // namespace mma
} // namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib