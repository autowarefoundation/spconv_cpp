#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m32n32k32m32n32k8A1_200_S1 {
namespace mma {
namespace mma_ns_miter {
struct MaskIGemmIterator {
  int k_idx;
  int filter_idx;
  const int& gemm_k_iterations;
  const int& RS;
  const uint32_t& mask;
  bool end;
  __forceinline__ __device__  MaskIGemmIterator(const int& gemm_k_iterations, const int& RS, const uint32_t& mask) : k_idx(0), filter_idx(0), gemm_k_iterations(gemm_k_iterations), RS(RS), mask(mask), end(false)  {
    
  }
  __forceinline__ __device__ void operator++()   {
    
    if (++filter_idx < RS){
        return;
    }
    filter_idx = 0;
    if (++k_idx < gemm_k_iterations){
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
} // namespace Simt_f32f32f32f32f32tnt_m32n32k32m32n32k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib