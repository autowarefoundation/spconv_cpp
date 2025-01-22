#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK {
namespace inpitera {
namespace gload {
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
          "  @p ld.global.L2::128B.u16 %0,[%2];\n"
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
} // namespace gload
} // namespace inpitera
} // namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib