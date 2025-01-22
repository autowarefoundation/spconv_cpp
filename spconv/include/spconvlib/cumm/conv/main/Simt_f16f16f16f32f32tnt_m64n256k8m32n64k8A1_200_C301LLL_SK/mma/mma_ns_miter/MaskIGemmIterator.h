#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK {
namespace mma {
namespace mma_ns_miter {
struct MaskIGemmIterator {
  int filter_idx;
  const int& gemm_k_iterations;
  const int& RS;
  const uint32_t& mask;
  bool end;
  __forceinline__ __device__  MaskIGemmIterator(const int& gemm_k_iterations, const int& RS, const uint32_t& mask) : filter_idx(0), gemm_k_iterations(gemm_k_iterations), RS(RS), mask(mask), end(false)  {
    
  }
  __forceinline__ __device__ void operator++()   {
    
    if (++filter_idx < RS){
        return;
    }
    end = true;
  }
  __forceinline__ __device__ bool valid()  const {
    
    return mask & (1u << filter_idx);
  }
};
} // namespace mma_ns_miter
} // namespace mma
} // namespace Simt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib