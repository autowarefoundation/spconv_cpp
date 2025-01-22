#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8s8s32f32tnt_m256n128k64m128n64k64A1T8816_200_C301LLL_SKD_S8 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<int8_t, 32768, 16> smem_A;
  tv::alignedarray<int8_t, 16384, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace Turing_s8s8s8s32f32tnt_m256n128k64m128n64k64A1T8816_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib