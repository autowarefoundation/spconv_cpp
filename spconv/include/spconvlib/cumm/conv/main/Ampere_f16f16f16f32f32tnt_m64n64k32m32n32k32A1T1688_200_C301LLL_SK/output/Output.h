#pragma once
#include <spconvlib/cumm/conv/main/Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK/output/out_ns_frag/OutFragIterTensorOp.h>
#include <spconvlib/cumm/conv/main/Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK/output/out_ns_warp/OutWarpTileIteratorTensorOpMixed.h>
#include <spconvlib/cumm/conv/main/Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK/output/out_ns_smem/OutSmemLoaderMixed.h>
#include <spconvlib/cumm/conv/main/Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK/out_iter/OutIterator.h>
#include <spconvlib/cumm/conv/main/Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK/out_iter_const/OutIterator.h>
#include <spconvlib/cumm/conv/main/Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK/out_smem_storage/OutputSmemStorage.h>
#include <spconvlib/cumm/conv/main/Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK/out_op/LinearCombination.h>
#include <spconvlib/cumm/conv/main/Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK/output/out_ns_apply/ApplyOutputOp.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK {
namespace output {
using FragIter = spconvlib::cumm::conv::main::Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::output::out_ns_frag::OutFragIterTensorOp;
using OutWarpIter = spconvlib::cumm::conv::main::Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::output::out_ns_warp::OutWarpTileIteratorTensorOpMixed;
using SmemLoader = spconvlib::cumm::conv::main::Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::output::out_ns_smem::OutSmemLoaderMixed;
using OutIter = spconvlib::cumm::conv::main::Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::out_iter::OutIterator;
using ConstOutIter = spconvlib::cumm::conv::main::Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::out_iter_const::OutIterator;
using OutputStorage = spconvlib::cumm::conv::main::Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::out_smem_storage::OutputSmemStorage;
using OutputOp = spconvlib::cumm::conv::main::Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::out_op::LinearCombination;
using ApplyOp = spconvlib::cumm::conv::main::Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::output::out_ns_apply::ApplyOutputOp;
struct Output {
  OutWarpIter warp_iter;
  SmemLoader smem_loader;
  __forceinline__ __device__  Output(OutputStorage* smem_storage, int thread_idx, int warp_idx_k, int warp_m, int warp_n, int lane_idx) : warp_iter(smem_storage->smem.data(), warp_idx_k * 2 + warp_m, warp_n, lane_idx), smem_loader(smem_storage->smem.data(), thread_idx)  {
    
  }
  __forceinline__ __device__ void run(OutputOp const& output_op, tv::array<float, 32, 0> const& accumulators, OutIter& out_iter, ConstOutIter& source_iter)   {
    
    if (!output_op.is_source_needed()){
      return run_no_source(output_op, accumulators, out_iter);
    }
    tv::array<tv::half_t, 8, 0> source_frag;
    source_frag.clear();
    FragIter out_acc_iter(accumulators.data());
    TV_PRAGMA_UNROLL
    for (int iter = 0; iter < 4; iter += 1){
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        tv::array<float, 8, 0> acc_frag;
        out_acc_iter.load(acc_frag);
        ++out_acc_iter;
        warp_iter.store(acc_frag);
        if (p < 1 - 1){
            warp_iter.add_pointer_offset(1152);
        }
      }
      if (1 > 1){
          warp_iter.add_pointer_offset(0);
      }
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        source_iter.load(source_frag);
        ++source_iter;
        tv::array<float, 8, 0> smem_frags[1];
        smem_loader.load(smem_frags[0]);
        if (p < 1 - 1){
            smem_loader.add_pointer_offset(1152);
        }
        else if (1 > 1){
            TV_PRAGMA_UNROLL
            for (int partk_idx = 1; partk_idx < 1; ++partk_idx){
                smem_loader.add_pointer_offset(1152);
                smem_loader.load(smem_frags[partk_idx]);
                tv::math::plus<tv::array<float, 8, 0>> accer;
                smem_frags[0] = accer(smem_frags[0], smem_frags[partk_idx]);
            }
            smem_loader.add_pointer_offset(0);
        }
        tv::array<tv::half_t, 8, 0> out_frag;
        ApplyOp::apply_output_operator(out_frag, output_op, smem_frags[0], source_frag);
        out_iter.store(out_frag);
        ++out_iter;
      }
      if (1 > 1){
          smem_loader.add_pointer_offset(0);
      }
    }
  }
  __forceinline__ __device__ void run_no_source(OutputOp const& output_op, tv::array<float, 32, 0> const& accumulators, OutIter& out_iter)   {
    
    FragIter out_acc_iter(accumulators.data());
    TV_PRAGMA_UNROLL
    for (int iter = 0; iter < 4; iter += 1){
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        tv::array<float, 8, 0> acc_frag;
        out_acc_iter.load(acc_frag);
        ++out_acc_iter;
        warp_iter.store(acc_frag);
        if (p < 1 - 1){
            warp_iter.add_pointer_offset(1152);
        }
      }
      if (1 > 1){
          warp_iter.add_pointer_offset(0);
      }
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        tv::array<float, 8, 0> smem_frags[1];
        smem_loader.load(smem_frags[0]);
        if (p < 1 - 1){
            smem_loader.add_pointer_offset(1152);
        }
        else if (1 > 1){
            TV_PRAGMA_UNROLL
            for (int partk_idx = 1; partk_idx < 1; ++partk_idx){
                smem_loader.add_pointer_offset(1152);
                smem_loader.load(smem_frags[partk_idx]);
                tv::math::plus<tv::array<float, 8, 0>> accer;
                smem_frags[0] = accer(smem_frags[0], smem_frags[partk_idx]);
            }
            smem_loader.add_pointer_offset(0);
        }
        tv::array<tv::half_t, 8, 0> out_frag;
        ApplyOp::apply_output_operator_no_source(out_frag, output_op, smem_frags[0]);
        out_iter.store(out_frag);
        ++out_iter;
      }
      if (1 > 1){
          smem_loader.add_pointer_offset(0);
      }
    }
  }
  __forceinline__ __device__ void run_self_reduce(OutputOp const& output_op, tv::array<float, 32, 0> const& accumulators, OutIter& out_iter)   {
    
    tv::array<tv::half_t, 8, 0> source_frag;
    source_frag.clear();
    FragIter out_acc_iter(accumulators.data());
    TV_PRAGMA_UNROLL
    for (int iter = 0; iter < 4; iter += 1){
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        tv::array<float, 8, 0> acc_frag;
        out_acc_iter.load(acc_frag);
        ++out_acc_iter;
        warp_iter.store(acc_frag);
        if (p < 1 - 1){
            warp_iter.add_pointer_offset(1152);
        }
      }
      if (1 > 1){
          warp_iter.add_pointer_offset(0);
      }
      __syncthreads();
      TV_PRAGMA_UNROLL
      for (int p = 0; p < 1; ++p){
        out_iter.load(source_frag);
        tv::array<float, 8, 0> smem_frags[1];
        smem_loader.load(smem_frags[0]);
        if (p < 1 - 1){
            smem_loader.add_pointer_offset(1152);
        }
        else if (1 > 1){
            TV_PRAGMA_UNROLL
            for (int partk_idx = 1; partk_idx < 1; ++partk_idx){
                smem_loader.add_pointer_offset(1152);
                smem_loader.load(smem_frags[partk_idx]);
                tv::math::plus<tv::array<float, 8, 0>> accer;
                smem_frags[0] = accer(smem_frags[0], smem_frags[partk_idx]);
            }
            smem_loader.add_pointer_offset(0);
        }
        tv::array<tv::half_t, 8, 0> out_frag;
        ApplyOp::apply_output_operator(out_frag, output_op, smem_frags[0], source_frag);
        out_iter.store(out_frag);
        ++out_iter;
      }
      if (1 > 1){
          smem_loader.add_pointer_offset(0);
      }
    }
  }
};
} // namespace output
} // namespace Ampere_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib