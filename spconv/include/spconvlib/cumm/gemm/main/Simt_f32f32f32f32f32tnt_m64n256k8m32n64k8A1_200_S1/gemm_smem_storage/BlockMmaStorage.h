#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m64n256k8m32n64k8A1_200_S1 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<float, 1088, 16> smem_A;
  tv::alignedarray<float, 4160, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Simt_f32f32f32f32f32tnt_m64n256k8m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib