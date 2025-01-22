#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8/inpitera/mask/Mask.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8/inpitera/gload/GlobalLoad.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8/inpitera/tmap/PitchLinearWarpRaked.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8/cp/ConvProblem.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8/itera_p/SparseParams.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8 {
namespace inpitera {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using Mask = spconvlib::cumm::conv::main::Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8::inpitera::mask::Mask;
using GlobalLoad = spconvlib::cumm::conv::main::Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8::inpitera::gload::GlobalLoad;
using ThreadMap = spconvlib::cumm::conv::main::Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8::inpitera::tmap::PitchLinearWarpRaked;
using ConvProblem = spconvlib::cumm::conv::main::cpTuring_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8::cp::ConvProblem;
using Params = spconvlib::cumm::conv::main::cpTuring_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8::itera_p::SparseParams;
struct ForwardDgradSparseIOIteratorV2Mask {
  Params & params_;
  ConvProblem const& problem_;
  const char * pointer_;
  int const* indice_ptr_;
  int reduce_channel_offset_;
  int reduce_channel_offset_backup_;
  Mask mask_reset_backup_;
  Mask mask_;
  int32_t indices_[1];
  __forceinline__ __device__  ForwardDgradSparseIOIteratorV2Mask(Params & params, ConvProblem const& problem_size, const int8_t * ptr, int thread_id, const tv::array<int, 2>& threadblock_offset) : params_(params), problem_(problem_size), indice_ptr_(params.indice_ptr_)  {
    
    auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
    int stride_offset_ = thread_offset[0];
    // update_indices();
    pointer_ = reinterpret_cast<const char *>(ptr + thread_offset[1]);
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
            for (int v = 0; v < 16; ++v){
                uint32_t pred = (stride_offset_ + s * 16 + ss) < problem_.N;
                // mask_[v] |= (pred << (s * 1 + ss));
                mask_.set_coord(pred, s, 0, ss, v);
            }
        }
    }
    mask_.mask_[0] = thread_offset[1] + 0 >= problem_.C ? 
        mask_.mask_[0] & 4294967294u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 1 >= problem_.C ? 
        mask_.mask_[0] & 4294967293u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 2 >= problem_.C ? 
        mask_.mask_[0] & 4294967291u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 3 >= problem_.C ? 
        mask_.mask_[0] & 4294967287u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 4 >= problem_.C ? 
        mask_.mask_[0] & 4294967279u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 5 >= problem_.C ? 
        mask_.mask_[0] & 4294967263u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 6 >= problem_.C ? 
        mask_.mask_[0] & 4294967231u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 7 >= problem_.C ? 
        mask_.mask_[0] & 4294967167u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 8 >= problem_.C ? 
        mask_.mask_[0] & 4294967039u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 9 >= problem_.C ? 
        mask_.mask_[0] & 4294966783u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 10 >= problem_.C ? 
        mask_.mask_[0] & 4294966271u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 11 >= problem_.C ? 
        mask_.mask_[0] & 4294965247u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 12 >= problem_.C ? 
        mask_.mask_[0] & 4294963199u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 13 >= problem_.C ? 
        mask_.mask_[0] & 4294959103u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 14 >= problem_.C ? 
        mask_.mask_[0] & 4294950911u : mask_.mask_[0];
    mask_.mask_[0] = thread_offset[1] + 15 >= problem_.C ? 
        mask_.mask_[0] & 4294934527u : mask_.mask_[0];
    // TV_PRAGMA_UNROLL
    // for (int v = 0; v < 16; ++v){
    //     mask_[v] = thread_offset[1] + v * 1 >= problem_.C ? 0 : mask_[v];
    // }
    mask_reset_backup_ = mask_;
  }
  __forceinline__ __device__ void update_indices()   {
    
    int mask_inds[1];
    uint32_t pred;
    pred = mask_.query_coord(0, 0, 0, 0);
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
            // if (mask_[0] & (1u << (s * 1 + ss)))
            if (mask_.query_coord(s, 0, ss, 0)){
                indices_[s * 1 + ss] = 
                indice_ptr_[mask_inds[s * 1 + ss]] * 
                    problem_.C * 1 ;
            }
        }
    }
  }
  __forceinline__ __device__ void clear_all_mask_if_not_pred(bool pred)   {
    
    return mask_.clear_all_mask_if_not_pred(pred);
  }
  __forceinline__ __device__ void clear_all_mask_if_pred(bool pred)   {
    
    return mask_.clear_all_mask_if_pred(pred);
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
    mask_.mask_[0] = reduce_channel_offset_ + 0 >= problem_.C ? 
        mask_.mask_[0] & 4294967294u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 1 >= problem_.C ? 
        mask_.mask_[0] & 4294967293u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 2 >= problem_.C ? 
        mask_.mask_[0] & 4294967291u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 3 >= problem_.C ? 
        mask_.mask_[0] & 4294967287u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 4 >= problem_.C ? 
        mask_.mask_[0] & 4294967279u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 5 >= problem_.C ? 
        mask_.mask_[0] & 4294967263u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 6 >= problem_.C ? 
        mask_.mask_[0] & 4294967231u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 7 >= problem_.C ? 
        mask_.mask_[0] & 4294967167u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 8 >= problem_.C ? 
        mask_.mask_[0] & 4294967039u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 9 >= problem_.C ? 
        mask_.mask_[0] & 4294966783u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 10 >= problem_.C ? 
        mask_.mask_[0] & 4294966271u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 11 >= problem_.C ? 
        mask_.mask_[0] & 4294965247u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 12 >= problem_.C ? 
        mask_.mask_[0] & 4294963199u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 13 >= problem_.C ? 
        mask_.mask_[0] & 4294959103u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 14 >= problem_.C ? 
        mask_.mask_[0] & 4294950911u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 15 >= problem_.C ? 
        mask_.mask_[0] & 4294934527u : mask_.mask_[0];
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
  __forceinline__ __device__ const tv::array<int8_t, 1> * get(int indice_offset)  const {
    
    return reinterpret_cast<const tv::array<int8_t, 1> *>( pointer_ + indice_offset);
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<int8_t, 16, 0>& frag, int32_t pointer_offset)   {
    
    frag.clear();
    tv::array<int8_t, 1> *frag_ptr = reinterpret_cast<tv::array<int8_t, 1> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s){
      TV_PRAGMA_UNROLL
      for (int c = 0; c < 1; ++c){
        TV_PRAGMA_UNROLL
        for (int ss = 0; ss < 1; ++ss){
          auto indice_offset = get_indice_offset(s, c, ss);
          TV_PRAGMA_UNROLL
          for (int v = 0; v < 16; ++v){
            // int mask_idx = s * 1 + 
            //     c * 1 + ss;
            int idx = s * 16 + 
                c * 16 + ss * 16 + v;
            bool valid = bool(mask_.query_coord(s, c, ss, v)) && (indice_offset >= 0);
            auto access_pointer = reinterpret_cast<const tv::array<int8_t, 1> *>(pointer_ + indice_offset + 
                c * 32) + v;
            // tv::gemm::global_load<tv::array<int8_t, 1>, sizeof(tv::array<int8_t, 1>)>(
            //    frag_ptr[idx], access_pointer, valid);
            GlobalLoad::run(frag_ptr[idx], access_pointer, valid);
          }
        }
      }
    }
  }
  __forceinline__ __device__ void load(tv::array<int8_t, 16, 0>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
};
} // namespace inpitera
} // namespace Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib