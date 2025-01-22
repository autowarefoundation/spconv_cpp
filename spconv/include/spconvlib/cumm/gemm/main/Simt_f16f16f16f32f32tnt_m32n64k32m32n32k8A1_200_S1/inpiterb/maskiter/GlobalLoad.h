#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1 {
namespace inpiterb {
namespace maskiter {
struct GlobalLoad {
  template <typename Frag>
  __forceinline__ __device__ static void run(Frag & frag, void const* ptr, bool pred)   {
    
    uint16_t* frag_ptr = reinterpret_cast<uint16_t*>(&frag);
    #if (CUDA_VERSION >= 11040 && (__CUDA_ARCH__ >= 750))
      asm volatile (
          "{\n"
          "  mov.u16 %0,%0;\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p,%1,0;\n"
          "  @p ld.global.u16 %0,[%2];\n"
          "}\n"
          : "+h"(frag_ptr[0])
          : "r"((int)pred), "l"(ptr)
      );
    #else
      asm volatile (
          "{\n"
          "  mov.u16 %0,%0;\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p,%1,0;\n"
          "  @p ld.global.u16 %0,[%2];\n"
          "}\n"
          : "+h"(frag_ptr[0])
          : "r"((int)pred), "l"(ptr)
      );
    #endif
  }
};
} // namespace maskiter
} // namespace inpiterb
} // namespace Simt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib