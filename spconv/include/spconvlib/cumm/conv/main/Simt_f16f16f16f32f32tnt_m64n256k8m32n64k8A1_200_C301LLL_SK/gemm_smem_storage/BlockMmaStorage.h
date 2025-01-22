#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<tv::half_t, 1152, 16> smem_A;
  tv::alignedarray<tv::half_t, 4224, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Simt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib