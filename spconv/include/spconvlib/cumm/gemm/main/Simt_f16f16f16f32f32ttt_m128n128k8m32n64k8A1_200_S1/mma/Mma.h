#pragma once
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/mma/mma_ns_miter/MaskIGemmIterator.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/mma/mma_ns_miterD/MaskIGemmIteratorMaskLoaderDynamic.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/mma/mma_ns_wa/WarpTileIterator.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/mma/mma_ns_wb/WarpTileIterator.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/mma/mma_ns_sa/SmemTileIteratorV2.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/mma/mma_ns_sb/SmemTileIteratorV2.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/gemm_smem_storage/BlockMmaStorage.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/inpitera/MaskTileIterator.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/inpiterb/MaskTileIterator.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/mma/mma_ns_wmma/WarpMmaSimt.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1 {
namespace mma {
using MaskIGemmIterator = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::mma::mma_ns_miter::MaskIGemmIterator;
using MaskIGemmIteratorDynamic = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::mma::mma_ns_miterD::MaskIGemmIteratorMaskLoaderDynamic;
using WarpIterA = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::mma::mma_ns_wa::WarpTileIterator;
using WarpIterB = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::mma::mma_ns_wb::WarpTileIterator;
using SmemIterA = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::mma::mma_ns_sa::SmemTileIteratorV2;
using SmemIterB = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::mma::mma_ns_sb::SmemTileIteratorV2;
using GemmStorage = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::gemm_smem_storage::BlockMmaStorage;
using InputIteratorA = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::inpitera::MaskTileIterator;
using InputIteratorB = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::inpiterb::MaskTileIterator;
using WarpMma = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::mma::mma_ns_wmma::WarpMmaSimt;
struct Mma {
  WarpIterA warp_iter_A;
  WarpIterB warp_iter_B;
  SmemIterA smem_iter_A;
  SmemIterB smem_iter_B;
  __forceinline__ __device__  Mma(GemmStorage* smem_storage, int thread_idx, int warp_idx_k, int warp_m, int warp_n, int lane_idx) : warp_iter_A(smem_storage->smem_A.data(), warp_idx_k, warp_m, lane_idx), warp_iter_B(smem_storage->smem_B.data(), warp_idx_k, warp_n, lane_idx), smem_iter_A(136, smem_storage->smem_A.data(), thread_idx), smem_iter_B(128, smem_storage->smem_B.data(), thread_idx)  {
    
  }
  __forceinline__ __device__ void operator()(int gemm_k_iterations, tv::array<float, 64, 0>& accumulators, InputIteratorA & input_iter_A, InputIteratorB & input_iter_B, tv::array<float, 64, 0> const& src_accumulators)   {
    
    accumulators = src_accumulators;
    tv::array<tv::half_t, 4, 0> input_frag_A;
    tv::array<tv::half_t, 4, 0> input_frag_B;
    input_frag_A.clear();
    input_frag_B.clear();
    input_iter_A.load(input_frag_A);
    input_iter_B.load(input_frag_B);
    ++input_iter_A;
    ++input_iter_B;
    // tv::print_fragment_meta_once<float, -1>(input_frag_A, "FirstInputA", blockIdx.z, gemm_k_iterations);
    // tv::print_fragment_meta_once<float, -1>(input_frag_B, "FirstInputB", blockIdx.z);
    smem_iter_A.store(input_frag_A);
    smem_iter_B.store(input_frag_B);
    __syncthreads();
    tv::array<tv::half_t, 8, 0> warp_frag_A[2];
    tv::array<tv::half_t, 8, 0> warp_frag_B[2];
    ++smem_iter_A;
    ++smem_iter_B;
    warp_iter_A.set_kgroup_index(0);
    warp_iter_B.set_kgroup_index(0);
    warp_iter_A.load(warp_frag_A[0]);
    warp_iter_B.load(warp_frag_B[0]);
    // tv::print_fragment_meta_once<float, -1>(warp_frag_A[0], "FirstWarpA", blockIdx.z, warp_frag_A[0].size());
    // tv::print_fragment_meta_once<float, -1>(warp_frag_B[0], "FirstWarpB", blockIdx.z);
    // if (blockIdx.z == 0){
    //     tv::print_fragment_once<float, 0, 8, -1>(warp_frag_A[0]);
    // }
    ++warp_iter_A;
    ++warp_iter_B;
    WarpMma warp_mma;
    int smem_write_stage_idx = 1;
    if (gemm_k_iterations <= 1) {
        input_iter_A.clear_mask();
        input_iter_B.clear_mask();
    }
    for (; gemm_k_iterations > 0; --gemm_k_iterations){
      TV_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < 8; ++warp_mma_k){
        if (warp_mma_k == 8 - 1) {
            // TODO
            // tv::printf2_once(gemm_k_iterations);
            smem_iter_A.store(input_frag_A);
            smem_iter_B.store(input_frag_B);
            __syncthreads();
            ++smem_iter_A;
            ++smem_iter_B;
            if (smem_write_stage_idx == 1) {
                smem_iter_A.tile_increment(-2);
                smem_iter_B.tile_increment(-2);
            } else {
                warp_iter_A.tile_increment(-2 *
                                        8);
                warp_iter_B.tile_increment(-2 *
                                        8);
            }
            smem_write_stage_idx ^= 1;
        }
        warp_iter_A.set_kgroup_index((warp_mma_k + 1) % 8);
        warp_iter_B.set_kgroup_index((warp_mma_k + 1) % 8);
        // if warp_mma_k is last, smem load next, warp load next too.
        warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);
        ++warp_iter_A;
        ++warp_iter_B;
        if (warp_mma_k == 0){
          input_iter_A.load(input_frag_A);
          input_iter_B.load(input_frag_B);
          // tv::print_fragment_meta_once<float, -1>(input_frag_A, "InputA", blockIdx.z);
          // tv::print_fragment_meta_once<float, -1>(input_frag_B, "InputB", blockIdx.z);
          ++input_iter_A;
          ++input_iter_B;
          if (gemm_k_iterations <= 2) {
              input_iter_A.clear_mask();
              input_iter_B.clear_mask();
          }
        }
        warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                warp_frag_B[warp_mma_k % 2], accumulators);
      }
    }
  }
};
} // namespace mma
} // namespace Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib