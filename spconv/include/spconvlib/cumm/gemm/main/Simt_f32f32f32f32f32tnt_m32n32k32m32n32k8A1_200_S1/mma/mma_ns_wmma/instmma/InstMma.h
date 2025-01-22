#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m32n32k32m32n32k8A1_200_S1 {
namespace mma {
namespace mma_ns_wmma {
namespace instmma {
struct InstMma {
  __forceinline__ __device__ void operator()(tv::array<float, 1, 0>& d, tv::array<float, 1, 0> const & a, tv::array<float, 1, 0> const & b, tv::array<float, 1, 0> const & c)   {
    
    d[0] = a[0] * b[0] + c[0];
  }
};
} // namespace instmma
} // namespace mma_ns_wmma
} // namespace mma
} // namespace Simt_f32f32f32f32f32tnt_m32n32k32m32n32k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib