#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_f16f16f16f16f16tnt_m64n64k32m32n32k32A1T1688_200_S1 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<tv::half_t, 4096, 16> smem_A;
  tv::alignedarray<tv::half_t, 4096, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Turing_f16f16f16f16f16tnt_m64n64k32m32n32k32A1T1688_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib