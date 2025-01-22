#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32ttt_m32n32k32m32n32k8A1_200_S1 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<float, 2112, 16> smem_A;
  tv::alignedarray<float, 2048, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Simt_f32f32f32f32f32ttt_m32n32k32m32n32k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib