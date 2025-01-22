#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_f16f16f16f32f32tnt_m32n128k64m32n32k32A1T1688_200_C301LLL_SK {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<tv::half_t, 4096, 16> smem_A;
  tv::alignedarray<tv::half_t, 16384, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Turing_f16f16f16f32f32tnt_m32n128k64m32n32k32A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib