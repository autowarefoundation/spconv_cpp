#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16ttt_m64n64k32m32n32k32A1T884_200_S1 {
namespace inpitera {
namespace maskiter {
struct GlobalLoad {
  template <typename Frag>
  __forceinline__ __device__ static void run(Frag & frag, void const* ptr, bool pred)   {
    
    uint32_t* frag_ptr = reinterpret_cast<uint32_t*>(&frag);
    #if (CUDA_VERSION >= 11040 && (__CUDA_ARCH__ >= 750))
      asm volatile (
          "{\n"
          "  mov.b32 %0,%0;\n"
          "  mov.b32 %1,%1;\n"
          "  mov.b32 %2,%2;\n"
          "  mov.b32 %3,%3;\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p,%4,0;\n"
          "  @p ld.global.v4.b32 {%0, %1, %2, %3},[%5];\n"
          "}\n"
          : "+r"(frag_ptr[0]), "+r"(frag_ptr[1]), "+r"(frag_ptr[2]), "+r"(frag_ptr[3])
          : "r"((int)pred), "l"(ptr)
      );
    #else
      asm volatile (
          "{\n"
          "  mov.b32 %0,%0;\n"
          "  mov.b32 %1,%1;\n"
          "  mov.b32 %2,%2;\n"
          "  mov.b32 %3,%3;\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p,%4,0;\n"
          "  @p ld.global.v4.b32 {%0, %1, %2, %3},[%5];\n"
          "}\n"
          : "+r"(frag_ptr[0]), "+r"(frag_ptr[1]), "+r"(frag_ptr[2]), "+r"(frag_ptr[3])
          : "r"((int)pred), "l"(ptr)
      );
    #endif
  }
};
} // namespace maskiter
} // namespace inpitera
} // namespace Volta_f16f16f16f16f16ttt_m64n64k32m32n32k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib