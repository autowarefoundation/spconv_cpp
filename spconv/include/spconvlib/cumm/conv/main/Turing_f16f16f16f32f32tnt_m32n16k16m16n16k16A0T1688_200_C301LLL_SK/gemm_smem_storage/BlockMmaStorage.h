#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<tv::half_t, 1024, 16> smem_A;
  tv::alignedarray<tv::half_t, 512, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib