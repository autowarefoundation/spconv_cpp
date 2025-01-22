#pragma once
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8/mma/mma_ns_miter/MaskIGemmIterator.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8/mma_miterd/MaskIGemmIteratorMaskLoaderDynamic.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8/mma/mma_ns_wa/WarpIteratorCrosswise.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8/mma/mma_ns_wb/WarpIteratorCrosswise.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8/mma/mma_ns_sa/SmemTileIterator.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8/mma/mma_ns_sb/SmemTileIterator.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8/gemm_smem_storage/BlockMmaStorage.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8/inpitera/ForwardDgradSparseIOIterator.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8/inpiterb/WeightIteratorDP4A.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8/mma/mma_ns_wmma/WarpMmaTuring.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8 {
namespace mma {
using MaskIGemmIterator = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8::mma::mma_ns_miter::MaskIGemmIterator;
using MaskIGemmIteratorDynamic = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8::mma_miterd::MaskIGemmIteratorMaskLoaderDynamic;
using WarpIterA = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8::mma::mma_ns_wa::WarpIteratorCrosswise;
using WarpIterB = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8::mma::mma_ns_wb::WarpIteratorCrosswise;
using SmemIterA = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8::mma::mma_ns_sa::SmemTileIterator;
using SmemIterB = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8::mma::mma_ns_sb::SmemTileIterator;
using GemmStorage = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8::gemm_smem_storage::BlockMmaStorage;
using InputIteratorA = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8::inpitera::ForwardDgradSparseIOIterator;
using InputIteratorB = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8::inpiterb::WeightIteratorDP4A;
using WarpMma = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8::mma::mma_ns_wmma::WarpMmaTuring;
struct Mma {
  WarpIterA warp_iter_A;
  WarpIterB warp_iter_B;
  SmemIterA smem_iter_A;
  SmemIterB smem_iter_B;
  __forceinline__ __device__  Mma(GemmStorage* smem_storage, int thread_idx, int warp_idx_k, int warp_m, int warp_n, int lane_idx) : warp_iter_A(smem_storage->smem_A.data(), warp_idx_k, warp_m, lane_idx), warp_iter_B(smem_storage->smem_B.data(), warp_idx_k, warp_n, lane_idx), smem_iter_A(64, smem_storage->smem_A.data(), thread_idx), smem_iter_B(32, smem_storage->smem_B.data(), thread_idx)  {
    
  }
  __forceinline__ __device__ void operator()(int gemm_k_iterations, tv::array<int32_t, 32, 0>& accumulators, InputIteratorA & input_iter_A, InputIteratorB & input_iter_B, tv::array<int32_t, 32, 0> const& src_accumulators, MaskIGemmIteratorDynamic& mask_iter, int RS)   {
    
    accumulators = src_accumulators;
    tv::array<int8_t, 32, 0> input_frag_A;
    tv::array<int8_t, 16, 0> input_frag_B;
    tv::array<int8_t, 16, 0> warp_frag_A[2];
    tv::array<int8_t, 16, 0> warp_frag_B[2];
    WarpMma warp_mma;
    int smem_write_stage_idx = 1;
    // mask = 0xffffffff;
    // find initial gemm index
    while (!mask_iter.valid()){
        ++mask_iter;
        input_iter_A.increment_filter();
        input_iter_B.increment_filter();
    }
    // now input iter point to a valid location, mask iter point to this location too.
    input_iter_A.update_indices();
    input_iter_A.load(input_frag_A);
    input_iter_B.load(input_frag_B);
    // move to next valid location.
    input_iter_A.increment_k();
    input_iter_B.increment_k();
    // TODO we should increment mask here to hidden increment compute time.
    smem_iter_A.store(input_frag_A);
    smem_iter_B.store(input_frag_B);
    __syncthreads();
    ++smem_iter_A;
    ++smem_iter_B;
    warp_iter_A.set_kgroup_index(0);
    warp_iter_B.set_kgroup_index(0);
    warp_iter_A.load(warp_frag_A[0]);
    warp_iter_B.load(warp_frag_B[0]);
    ++warp_iter_A;
    ++warp_iter_B;
    while (!mask_iter.end){
        // tv::printf2_once(mask_iter.k_idx, mask_iter.filter_idx);
        for (int i = 0; i < gemm_k_iterations; ++i){
            TV_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < 2; ++warp_mma_k){
                if (warp_mma_k == 2 - 1) {
                    // save to S1
                    smem_iter_A.store(input_frag_A);
                    smem_iter_B.store(input_frag_B);
                    __syncthreads();
                    ++smem_iter_A;
                    ++smem_iter_B;
                    // SMEM double buffer
                    if (smem_write_stage_idx == 1) {
                        // back to S0
                        smem_iter_A.tile_increment(-2);
                        smem_iter_B.tile_increment(-2);
                    } else {
                        // 
                        warp_iter_A.tile_increment(-2 *
                                                2);
                        warp_iter_B.tile_increment(-2 *
                                                2);
                    }
                    smem_write_stage_idx ^= 1;
                }
                warp_iter_A.set_kgroup_index((warp_mma_k + 1) % 2);
                warp_iter_B.set_kgroup_index((warp_mma_k + 1) % 2);
                warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);
                ++warp_iter_A;
                ++warp_iter_B;
                // load next input frag
                // to hide long input latency
                // by a whole wmma operation
                if (warp_mma_k == 0){
                    // 01 001
                    // here input iter point to next location of current
                    // mask iter (may be invalid), we need to increment to
                    // find a valid location.
                    if (i == gemm_k_iterations - 1){
                        input_iter_A.reset_k();
                        input_iter_B.reset_k();
                        ++mask_iter;
                        input_iter_A.increment_filter();
                        input_iter_B.increment_filter();
                        while (!mask_iter.valid() && !mask_iter.end){
                            ++mask_iter;
                            input_iter_A.increment_filter();
                            input_iter_B.increment_filter();
                        }
                        // load next indices
                        // TODO why do we need 20 more registers when use if?
                        input_iter_A.clear_all_mask_if_pred(mask_iter.end);
                        input_iter_B.clear_all_mask_if_pred(mask_iter.end);
                        input_iter_A.update_indices();
                    }
                    input_iter_A.load(input_frag_A);
                    input_iter_B.load(input_frag_B);
                    input_iter_A.increment_k();
                    input_iter_B.increment_k();
                }
                warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                        warp_frag_B[warp_mma_k % 2], accumulators);
            }
        }
    }
  }
};
} // namespace mma
} // namespace Turing_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T8816_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib