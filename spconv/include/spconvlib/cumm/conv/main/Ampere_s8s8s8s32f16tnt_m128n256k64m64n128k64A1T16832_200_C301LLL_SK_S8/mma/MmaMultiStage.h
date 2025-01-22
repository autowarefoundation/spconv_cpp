#pragma once
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/mma_ns_miter/MaskIGemmIterator.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma_miterd/MaskIGemmIteratorMaskLoaderDynamic.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/mma_ns_wa/WarpIteratorCrosswise.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/mma_ns_wb/WarpIteratorCrosswise.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/mma_ns_sa/SmemTileIterator.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/mma_ns_sb/SmemTileIterator.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/gemm_smem_storage/BlockMmaStorage.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/cpasync_group/CpAsyncGroup.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/async_cp_iter_global_A/AsyncCopyIteration.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/async_cp_iter_global_B/AsyncCopyIteration.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/async_cp_iter_0_A/AsyncCopyIteration.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/async_cp_iter_0_B/AsyncCopyIteration.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/async_cp_iter_1_A/AsyncCopyIteration.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/async_cp_iter_1_B/AsyncCopyIteration.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/inpitera/ForwardDgradSparseIOIterator.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/inpiterb/WeightIteratorDP4A.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8/mma/mma_ns_wmma/WarpMmaTuring.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8 {
namespace mma {
using MaskIGemmIterator = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::mma_ns_miter::MaskIGemmIterator;
using MaskIGemmIteratorDynamic = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma_miterd::MaskIGemmIteratorMaskLoaderDynamic;
using WarpIterA = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::mma_ns_wa::WarpIteratorCrosswise;
using WarpIterB = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::mma_ns_wb::WarpIteratorCrosswise;
using SmemIterA = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::mma_ns_sa::SmemTileIterator;
using SmemIterB = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::mma_ns_sb::SmemTileIterator;
using GemmStorage = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::gemm_smem_storage::BlockMmaStorage;
using CpAsyncGroup = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::cpasync_group::CpAsyncGroup;
using GlobalAsyncCopyIter_A = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::async_cp_iter_global_A::AsyncCopyIteration;
using GlobalAsyncCopyIter_B = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::async_cp_iter_global_B::AsyncCopyIteration;
using AsyncCopyIter_0_A = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::async_cp_iter_0_A::AsyncCopyIteration;
using AsyncCopyIter_0_B = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::async_cp_iter_0_B::AsyncCopyIteration;
using AsyncCopyIter_1_A = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::async_cp_iter_1_A::AsyncCopyIteration;
using AsyncCopyIter_1_B = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::async_cp_iter_1_B::AsyncCopyIteration;
using InputIteratorA = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::inpitera::ForwardDgradSparseIOIterator;
using InputIteratorB = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::inpiterb::WeightIteratorDP4A;
using WarpMma = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8::mma::mma_ns_wmma::WarpMmaTuring;
struct MmaMultiStage {
  WarpIterA warp_iter_A;
  WarpIterB warp_iter_B;
  SmemIterA smem_iter_A;
  SmemIterB smem_iter_B;
  __forceinline__ __device__  MmaMultiStage(GemmStorage* smem_storage, int thread_idx, int warp_idx_k, int warp_m, int warp_n, int lane_idx) : warp_iter_A(smem_storage->smem_A.data(), warp_idx_k, warp_m, lane_idx), warp_iter_B(smem_storage->smem_B.data(), warp_idx_k, warp_n, lane_idx), smem_iter_A(128, smem_storage->smem_A.data(), thread_idx), smem_iter_B(256, smem_storage->smem_B.data(), thread_idx)  {
    
  }
  __forceinline__ __device__ void copy_tiles_and_advance(InputIteratorA & input_iter_A, InputIteratorB & input_iter_B, const int & group_idx)   {
    
    #if (defined(DEBUG_MMA_MS_DOWNFALL_A) || defined(DEBUG_MMA_MS_DOWNFALL_B))
                      if (group_idx == 1){
      #ifdef DEBUG_MMA_MS_DOWNFALL_A
                      tv::array<int8_t, 64, 0> input_frag_A;
      #ifndef DEBUG_MMA_MS_NOT_READ_INPUT_A
                          input_iter_A.load(input_frag_A);
      #endif
      #ifndef DEBUG_MMA_MS_NOT_WRITE_SMEM_A
                          smem_iter_A.store(input_frag_A);
      #endif
      #endif
      #ifdef DEBUG_MMA_MS_DOWNFALL_B
                      tv::array<int8_t, 128, 0> input_frag_B;
      #ifndef DEBUG_MMA_MS_NOT_READ_INPUT_B
                          input_iter_B.load(input_frag_B);
      #endif
      #ifndef DEBUG_MMA_MS_NOT_WRITE_SMEM_B
                          smem_iter_B.store(input_frag_B);
      #endif
      #endif
                      }
    #endif
                        if(group_idx == 0){
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_A) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_A))
                            AsyncCopyIter_0_A::do_copy_zfill(input_iter_A, smem_iter_A);
    #endif
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_B) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_B))
                            AsyncCopyIter_0_B::do_copy_zfill(input_iter_B, smem_iter_B);
    #endif
                            return;
                        }
                        if(group_idx == 1){
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_A) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_A))
                            AsyncCopyIter_1_A::do_copy_zfill(input_iter_A, smem_iter_A);
    #endif
    #if (!defined(DEBUG_MMA_MS_DOWNFALL_B) && !defined(DEBUG_MMA_MS_NOT_WRITE_SMEM_B))
                            AsyncCopyIter_1_B::do_copy_zfill(input_iter_B, smem_iter_B);
    #endif
                            return;
                        }
  }
  __forceinline__ __device__ void operator()(const int& gemm_k_iterations, tv::array<int32_t, 256, 0>& accumulators, InputIteratorA & input_iter_A, InputIteratorB & input_iter_B, tv::array<int32_t, 256, 0> const& src_accumulators, uint32_t mask, const int& RS)   {
    
    accumulators = src_accumulators;
    tv::array<int8_t, 64, 0> warp_frag_A[2];
    tv::array<int8_t, 128, 0> warp_frag_B[2];
    WarpMma warp_mma;
    int smem_write_stage_idx = 1;
    int smem_read_stage_idx = 0;
    MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
    int local_gemm_k_iterations = gemm_k_iterations;
    while(!mask_iter.valid()){
        ++mask_iter;
        input_iter_A.increment_filter();
        input_iter_B.increment_filter();
    }
    input_iter_A.update_indices();
    TV_PRAGMA_UNROLL
    for (int stage=0; stage < 1; ++stage){
        GlobalAsyncCopyIter_A::do_copy_zfill(input_iter_A, smem_iter_A);
        GlobalAsyncCopyIter_B::do_copy_zfill(input_iter_B, smem_iter_B);
        CpAsyncGroup::make_fence();
        input_iter_A.increment_k();
        input_iter_B.increment_k();
        ++smem_iter_A;
        ++smem_iter_B;
        --local_gemm_k_iterations;
        if (!mask_iter.end && local_gemm_k_iterations == 0){
            ++mask_iter;
            input_iter_A.reset_k();
            input_iter_B.reset_k();
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
            while (!mask_iter.valid() && !mask_iter.end){
                ++mask_iter;
                input_iter_A.increment_filter();
                input_iter_B.increment_filter();
            }
            input_iter_A.clear_all_mask_if_pred(mask_iter.end);
            input_iter_B.clear_all_mask_if_pred(mask_iter.end);
            input_iter_A.update_indices();
            local_gemm_k_iterations = gemm_k_iterations;
        }
    }
    CpAsyncGroup::wait_final_group();
    __syncthreads();
    warp_iter_A.set_kgroup_index(0);
    warp_iter_B.set_kgroup_index(0);
    warp_iter_A.load(warp_frag_A[0]);
    warp_iter_B.load(warp_frag_B[0]);
    ++warp_iter_A;
    ++warp_iter_B;
    while (local_gemm_k_iterations != -1){
        TV_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < 2; ++warp_mma_k){
            warp_iter_A.set_kgroup_index((warp_mma_k + 1) % 2);
            warp_iter_B.set_kgroup_index((warp_mma_k + 1) % 2);
            warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
            warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);
            ++warp_iter_A;
            ++warp_iter_B;
            warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                warp_frag_B[warp_mma_k % 2], accumulators);
            if (warp_mma_k < 1)
                copy_tiles_and_advance(input_iter_A, input_iter_B, warp_mma_k);
            if (warp_mma_k + 2 == 2){
                copy_tiles_and_advance(input_iter_A, input_iter_B, 1);
                CpAsyncGroup::make_fence();
                // do chores before wait
                ++smem_iter_A;
                ++smem_iter_B;
                input_iter_A.increment_k();
                input_iter_B.increment_k();
                --local_gemm_k_iterations;
                if (!mask_iter.end && local_gemm_k_iterations == 0){
                    ++mask_iter;
                    input_iter_A.reset_k();
                    input_iter_B.reset_k();
                    input_iter_A.increment_filter();
                    input_iter_B.increment_filter();
                    while (!mask_iter.valid() && !mask_iter.end){
                        ++mask_iter;
                        input_iter_A.increment_filter();
                        input_iter_B.increment_filter();
                    }
                    input_iter_A.clear_all_mask_if_pred(mask_iter.end);
                    input_iter_B.clear_all_mask_if_pred(mask_iter.end);
                    input_iter_A.update_indices();
                    local_gemm_k_iterations = gemm_k_iterations;
                }
                if (smem_write_stage_idx == 1) {
                    smem_iter_A.tile_increment(-2);
                    smem_iter_B.tile_increment(-2);
                    smem_write_stage_idx = 0;
                } else
                    ++smem_write_stage_idx;
                if (smem_read_stage_idx == 1) {
                    warp_iter_A.tile_increment(-2 *
                                            2);
                    warp_iter_B.tile_increment(-2 *
                                            2);
                    smem_read_stage_idx = 0;
                } else
                    ++smem_read_stage_idx;
                // finish chores
                CpAsyncGroup::wait_final_group();
                __syncthreads();
            }
        }
    }
  }
};
} // namespace mma
} // namespace Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_200_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib