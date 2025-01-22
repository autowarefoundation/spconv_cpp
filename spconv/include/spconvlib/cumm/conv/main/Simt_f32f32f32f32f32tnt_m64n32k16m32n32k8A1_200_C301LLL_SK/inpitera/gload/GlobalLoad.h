#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK {
namespace inpitera {
namespace gload {
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
          "  @p ld.global.L2::128B.b32 %0,[%2];\n"
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
} // namespace gload
} // namespace inpitera
} // namespace Simt_f32f32f32f32f32tnt_m64n32k16m32n32k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib