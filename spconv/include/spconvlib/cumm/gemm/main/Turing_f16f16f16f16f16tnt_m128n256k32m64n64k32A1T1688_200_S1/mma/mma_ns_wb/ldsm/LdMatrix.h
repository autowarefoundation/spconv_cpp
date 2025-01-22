#pragma once
#include <tensorview/gemm/arch/memory_sm75.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_f16f16f16f16f16tnt_m128n256k32m64n64k32A1T1688_200_S1 {
namespace mma {
namespace mma_ns_wb {
namespace ldsm {
struct LdMatrix {
  __forceinline__ __device__ static void run(tv::array<unsigned, 4, 0> & D, void const* ptr)   {
    
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
      unsigned addr = tv::gemm::get_smem_pointer(ptr);
      int x, y, z, w;
      asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
      reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
    #else
      assert(0);
    #endif
  }
};
} // namespace ldsm
} // namespace mma_ns_wb
} // namespace mma
} // namespace Turing_f16f16f16f16f16tnt_m128n256k32m64n64k32A1T1688_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib