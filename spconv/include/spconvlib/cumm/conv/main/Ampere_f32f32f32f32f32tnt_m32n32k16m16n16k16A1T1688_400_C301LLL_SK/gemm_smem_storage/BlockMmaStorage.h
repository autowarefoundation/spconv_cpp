#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<float, 2048, 16> smem_A;
  tv::alignedarray<float, 2048, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib