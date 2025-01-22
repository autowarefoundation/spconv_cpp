#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T884_200_S1 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<tv::half_t, 8192, 16> smem_A;
  tv::alignedarray<tv::half_t, 16384, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Volta_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib