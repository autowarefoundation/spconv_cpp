#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f32tnt_m128n64k32m64n32k32A1T16816_400_C301LLL_SK_S8 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<int8_t, 16384, 16> smem_A;
  tv::alignedarray<int8_t, 8192, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Ampere_s8s8s8s32f32tnt_m128n64k32m64n32k32A1T16816_400_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib