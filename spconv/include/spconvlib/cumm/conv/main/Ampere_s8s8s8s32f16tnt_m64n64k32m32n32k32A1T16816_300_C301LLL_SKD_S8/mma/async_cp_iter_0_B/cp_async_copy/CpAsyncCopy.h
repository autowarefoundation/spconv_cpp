#pragma once

            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                #define CUDA_CP_ASYNC_ACTIVATED 1
            #else
                #define CUDA_CP_ASYNC_ACTIVATED 0
            #endif
            #if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
                (__CUDACC_VER_MAJOR__ > 11)) &&                                  \
                defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && \
                ! (defined(__clang__) && defined(__CUDA__))
                #define CUMM_ENABLE_L2_PREFETCH 1
            #else
                #define CUMM_ENABLE_L2_PREFETCH 0
            #endif
        
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f16tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8 {
namespace mma {
namespace async_cp_iter_0_B {
namespace cp_async_copy {
struct CpAsyncCopy {
  __forceinline__ __device__ static void copy(void* dest_smem, const void* src_global, bool pred_guard = true)   {
    
    unsigned smem_addr = tv::gemm::get_smem_pointer(dest_smem);
    #if (CUDA_CP_ASYNC_ACTIVATED)
                          asm volatile(
                              "{\n"
                              "  .reg .pred p;\n"
                              "  setp.ne.b32 p, %0, 0;\n"
      #if CUMM_ENABLE_L2_PREFETCH
                              "  @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
      #else
                              "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      #endif
                              "}\n" ::"r"((int)pred_guard), "r"(smem_addr), "l"(src_global), "n"(16));
    #else
      assert(0);
    #endif
  }
  __forceinline__ __device__ static void copy_zfill(void* dest_smem, const void* src_global, bool pred_guard = true)   {
    
    
    #if (CUDA_CP_ASYNC_ACTIVATED)
      unsigned smem_addr = tv::gemm::get_smem_pointer(dest_smem);
      unsigned real_size = (pred_guard ? 16 : 0);
                          asm volatile(
      #if CUMM_ENABLE_L2_PREFETCH
                              "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n"
      #else
                              "cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
      #endif
                               ::"r"(smem_addr), "l"(src_global), "n"(16), "r"(real_size));
    #else
      assert(0);
    #endif
  }
};
} // namespace cp_async_copy
} // namespace async_cp_iter_0_B
} // namespace mma
} // namespace Ampere_s8s8s8s32f16tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib