#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m32n512k8m32n64k8A1_200_S1 {
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
          "  .reg .pred p;\n"
          "  setp.ne.b32 p,%1,0;\n"
          "  @p ld.global.b32 %0,[%2];\n"
          "}\n"
          : "+r"(frag_ptr[0])
          : "r"((int)pred), "l"(ptr)
      );
    #else
      asm volatile (
          "{\n"
          "  mov.b32 %0,%0;\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p,%1,0;\n"
          "  @p ld.global.b32 %0,[%2];\n"
          "}\n"
          : "+r"(frag_ptr[0])
          : "r"((int)pred), "l"(ptr)
      );
    #endif
  }
};
} // namespace maskiter
} // namespace inpitera
} // namespace Simt_f32f32f32f32f32tnt_m32n512k8m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib