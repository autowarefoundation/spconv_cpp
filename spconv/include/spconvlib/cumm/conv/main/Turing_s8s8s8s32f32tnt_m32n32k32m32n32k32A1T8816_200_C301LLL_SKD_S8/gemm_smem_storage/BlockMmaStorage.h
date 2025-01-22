#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8s8s32f32tnt_m32n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<int8_t, 2048, 16> smem_A;
  tv::alignedarray<int8_t, 2048, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Turing_s8s8s8s32f32tnt_m32n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib