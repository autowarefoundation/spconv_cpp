#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<tv::half_t, 2176, 16> smem_A;
  tv::alignedarray<tv::half_t, 4096, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Simt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib