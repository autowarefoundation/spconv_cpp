#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m64n64k8m32n32k8A1_200_C301LLL_SK {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<float, 1088, 16> smem_A;
  tv::alignedarray<float, 1088, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Simt_f32f32f32f32f32tnt_m64n64k8m32n32k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib