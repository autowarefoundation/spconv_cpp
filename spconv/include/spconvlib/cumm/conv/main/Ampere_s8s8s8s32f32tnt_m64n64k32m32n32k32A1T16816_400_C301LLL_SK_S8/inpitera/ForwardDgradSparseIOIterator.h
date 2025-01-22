#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8/inpitera/mask/Mask.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8/inpitera/gload/GlobalLoad.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8/inpitera/tmap/PitchLinearWarpRaked.h>
#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8/cp/ConvProblem.h>
#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8/itera_p/SparseParams.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8 {
namespace inpitera {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using Mask = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8::inpitera::mask::Mask;
using GlobalLoad = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8::inpitera::gload::GlobalLoad;
using ThreadMap = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8::inpitera::tmap::PitchLinearWarpRaked;
using ConvProblem = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8::cp::ConvProblem;
using Params = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8::itera_p::SparseParams;
struct ForwardDgradSparseIOIterator {
  Params & params_;
  ConvProblem const& problem_;
  const char * pointer_;
  int const* indice_ptr_;
  int reduce_channel_offset_;
  int reduce_channel_offset_backup_;
  tv::array<uint32_t, 1> mask_reset_backup_;
  tv::array<uint32_t, 1> mask_;
  int32_t indices_[1];
  __forceinline__ __device__  ForwardDgradSparseIOIterator(Params & params, ConvProblem const& problem_size, const int8_t * ptr, int thread_id, const tv::array<int, 2>& threadblock_offset) : params_(params), problem_(problem_size), indice_ptr_(params.indice_ptr_)  {
    
    auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
    int stride_offset_ = thread_offset[0];
    // update_indices();
    pointer_ = reinterpret_cast<const char *>(ptr + thread_offset[1]);
    // std::uintptr_t access_pointer_num = reinterpret_cast<std::uintptr_t>(pointer_);
    // if (access_pointer_num % 16 != 0){
    //     tv::printf2_block_once("BBBBBBBBBBBBBBBBSFASF");
    // }
    // origin_pointer_ = pointer_;
    params.mask_argsort_ptr_ += stride_offset_;
    // mask_ = 0;
    mask_.clear();
    reduce_channel_offset_ = thread_offset[1];
    reduce_channel_offset_backup_ = thread_offset[1];
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s){
        TV_PRAGMA_UNROLL
        for (int ss = 0; ss < 1; ++ss){
            TV_PRAGMA_UNROLL
            for (int v = 0; v < 1; ++v){
                uint32_t pred = (stride_offset_ + s * 16 + ss) < problem_.N;
                mask_[v] |= (pred << (s * 1 + ss));
            }
        }
    }
    TV_PRAGMA_UNROLL
    for (int v = 0; v < 1; ++v){
        mask_[v] = thread_offset[1] + v * 16 >= problem_.C ? 0 : mask_[v];
    }
    mask_reset_backup_ = mask_;
  }
  __forceinline__ __device__ void update_indices()   {
    
    int mask_inds[1];
    uint32_t pred;
    pred = mask_[0] & (1u << (0 + 0));
    asm volatile (
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p,%1,0;\n"
        "  @p ld.global.b32 %0,[%2];\n"
        "}\n"
        : "=r"(mask_inds[0])
        : "r"(pred), "l"(params_.mask_argsort_ptr_)
    );
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s){
        TV_PRAGMA_UNROLL
        for (int ss = 0; ss < 1; ++ss){
            if (mask_[0] & (1u << (s * 1 + ss))){
                indices_[s * 1 + ss] = 
                indice_ptr_[mask_inds[s * 1 + ss]] * 
                    problem_.C * 1 ;
            }
        }
    }
  }
  __forceinline__ __device__ void clear_mask_if_not_pred(bool pred, int v)   {
    
    mask_[v] = pred ? mask_[v] : 0;
  }
  __forceinline__ __device__ void clear_all_mask_if_not_pred(bool pred)   {
    
    TV_PRAGMA_UNROLL
    for (int v = 0; v < 1; ++v){
        mask_[v] = pred ? mask_[v] : 0;
    }
  }
  __forceinline__ __device__ void clear_all_mask_if_pred(bool pred)   {
    
    TV_PRAGMA_UNROLL
    for (int v = 0; v < 1; ++v){
        mask_[v] = pred ? 0:  mask_[v];
    }
  }
  __forceinline__ __device__ void clear_mask_if_pred(bool pred, int v)   {
    
    mask_[v] = pred ? 0 : mask_[v];
  }
  __forceinline__ __device__ void operator++()   {
    
  }
  __forceinline__ __device__ void increment_no_clear_mask()   {
    
  }
  __forceinline__ __device__ void clear_mask_if_batch_unbound()   {
    
  }
  __forceinline__ __device__ void operator+=(int num)   {
    
  }
  __forceinline__ __device__ void increment_k()   {
    
    pointer_ += params_.inc_c_next;
    reduce_channel_offset_ += params_.filter_c_delta;
    TV_PRAGMA_UNROLL
    for (int v = 0; v < 1; ++v){
        clear_mask_if_pred(reduce_channel_offset_ + v * 16 >= problem_.C, v);
    }
  }
  __forceinline__ __device__ void increment_filter()   {
    
    indice_ptr_ += problem_.N;
  }
  __forceinline__ __device__ void increment_filter(int num)   {
    
    indice_ptr_ += problem_.N * num;
  }
  __forceinline__ __device__ void reset_k()   {
    
    pointer_ += params_.inc_c_reset;
    mask_ = mask_reset_backup_;
    reduce_channel_offset_ = reduce_channel_offset_backup_;
  }
  __forceinline__ __device__ int get_indice_offset(int stride, int contig, int ss)  const {
    
    return indices_[stride * 1 + ss];
  }
  __forceinline__ __device__ const tv::alignedarray<int4, 1, 16> * get(int indice_offset)  const {
    
    return reinterpret_cast<const tv::alignedarray<int4, 1, 16> *>( pointer_ + indice_offset);
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<int8_t, 16, 0>& frag, int32_t pointer_offset)   {
    
    frag.clear();
    tv::alignedarray<int4, 1, 16> *frag_ptr = reinterpret_cast<tv::alignedarray<int4, 1, 16> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s){
      TV_PRAGMA_UNROLL
      for (int c = 0; c < 1; ++c){
        TV_PRAGMA_UNROLL
        for (int ss = 0; ss < 1; ++ss){
          TV_PRAGMA_UNROLL
          for (int v = 0; v < 1; ++v){
            int mask_idx = s * 1 + 
                c * 1 + ss;
            int idx = s * 1 + 
                c * 1 + ss * 1 + v;
            auto indice_offset = get_indice_offset(s, c, ss);
            bool valid = bool(mask_[v] & (1u << mask_idx)) && (indice_offset >= 0);
            auto access_pointer = reinterpret_cast<const tv::alignedarray<int4, 1, 16> *>(pointer_ + indice_offset + 
                c * 32) + v;
            // std::uintptr_t access_pointer_num = reinterpret_cast<std::uintptr_t>(access_pointer);
            // std::uintptr_t access_pointer_num2 = reinterpret_cast<std::uintptr_t>(pointer_);
            // if (access_pointer_num % 16 != 0 && valid){
            //     tv::printf2(valid, s, access_pointer_num2 % 16, indice_offset, indice_offset%16, "AS", indices_[0], "A", blockIdx.x, blockIdx.y, blockIdx.z);
            // }
            // tv::gemm::global_load<tv::alignedarray<int4, 1, 16>, sizeof(tv::alignedarray<int4, 1, 16>)>(
            //    frag_ptr[idx], access_pointer, valid);
            GlobalLoad::run(frag_ptr[idx], access_pointer, valid);
          }
        }
      }
    }
  }
  __forceinline__ __device__ const tv::alignedarray<int4, 1, 16> * load_ptr_with_param(int s, int c, bool& valid_ref)   {
    
    int mask_idx = s * 1 + 
        c * 1 + 0;
    auto indice_offset = get_indice_offset(s, c, 0);
    valid_ref = bool(mask_[0] & (1u << mask_idx)) && (indice_offset >= 0);
    auto access_pointer = reinterpret_cast<const tv::alignedarray<int4, 1, 16> *>(pointer_ + indice_offset + 
        c * 32) + 0;
    return access_pointer;
  }
  __forceinline__ __device__ void load(tv::array<int8_t, 16, 0>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
};
} // namespace inpitera
} // namespace Ampere_s8s8s8s32f32tnt_m64n64k32m32n32k32A1T16816_400_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib