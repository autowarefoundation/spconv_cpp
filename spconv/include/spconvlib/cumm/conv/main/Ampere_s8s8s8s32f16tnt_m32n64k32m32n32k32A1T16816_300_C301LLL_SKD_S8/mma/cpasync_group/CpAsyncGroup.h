#pragma once

            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                #define CUDA_CP_ASYNC_ACTIVATED 1
            #else
                #define CUDA_CP_ASYNC_ACTIVATED 0
            #endif
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8 {
namespace mma {
namespace cpasync_group {
struct CpAsyncGroup {
  __forceinline__ __device__ static void make_fence()   {
    
    #if (CUDA_CP_ASYNC_ACTIVATED)
      asm volatile("cp.async.commit_group;\n" ::);
    #else
      assert(0);
    #endif
  }
  __forceinline__ __device__ static void wait_final_group()   {
    
    #if (CUDA_CP_ASYNC_ACTIVATED)
      asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
    #else
      assert(0);
    #endif
  }
  __forceinline__ __device__ static void wait_all()   {
    
    #if (CUDA_CP_ASYNC_ACTIVATED)
      asm volatile("cp.async.wait_all;\n" ::);
    #else
      assert(0);
    #endif
  }
};
} // namespace cpasync_group
} // namespace mma
} // namespace Ampere_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib