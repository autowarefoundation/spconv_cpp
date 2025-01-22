#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8/inpiterb/gload/GlobalLoad.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8/inpiterb/tmap/PitchLinearWarpRaked.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8/cp/ConvProblem.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8/iterb_p/WeightOptParams.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8/lb/TensorGeneric.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8 {
namespace inpiterb {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using GlobalLoad = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::inpiterb::gload::GlobalLoad;
using ThreadMap = spconvlib::cumm::conv::main::Turing_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::inpiterb::tmap::PitchLinearWarpRaked;
using ConvProblem = spconvlib::cumm::conv::main::cpTuring_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::cp::ConvProblem;
using Params = spconvlib::cumm::conv::main::cpTuring_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::iterb_p::WeightOptParams;
using Layout = spconvlib::cumm::conv::main::cpTuring_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8::lb::TensorGeneric;
struct WeightIteratorDP4A {
  Params const& params_;
  ConvProblem const& problem_size_;
  const char * pointer_;
  int reduce_channel_offset_;
  int reduce_channel_offset_backup_;
  tv::array<uint32_t, 1> mask_backup_;
  tv::array<uint32_t, 1> mask_;
  __forceinline__ __device__  WeightIteratorDP4A(Params const& params, ConvProblem const& problem_size, const int8_t * ptr, int thread_id, const tv::array<int, 2>& threadblock_offset) : params_(params), problem_size_(problem_size), pointer_(reinterpret_cast<const char *>(ptr))  {
    
    auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
    mask_.clear();
    reduce_channel_offset_ = thread_offset[1];
    reduce_channel_offset_backup_ = thread_offset[1];
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 2; ++s){
      TV_PRAGMA_UNROLL
      for (int c = 0; c < 1; ++c){
        TV_PRAGMA_UNROLL
        for (int ss = 0; ss < 1; ++ss){
          uint32_t pred = thread_offset[0] + s * 16 + ss < problem_size.K;
          TV_PRAGMA_UNROLL
          for (int v = 0; v < 1; ++v){
            mask_[v] |= (pred << (s * 1 + c * 1 + ss));
          }
        }
      }
    }
    TV_PRAGMA_UNROLL
    for (int v = 0; v < 1; ++v){
        mask_[v] = thread_offset[1] + v * 16 >= problem_size.C ? 0 : mask_[v];
    }
    pointer_ += (thread_offset[0] * params.layout.strides[0] + thread_offset[1]) * 8 / 8;
    mask_backup_ = mask_;
  }
  __forceinline__ __device__ void operator++()   {
    
  }
  __forceinline__ __device__ void increment_k()   {
    
    pointer_ += params_.inc_c;
    reduce_channel_offset_ += params_.filter_c_delta;
    TV_PRAGMA_UNROLL
    for (int v = 0; v < 1; ++v){
        clear_mask_if_pred(reduce_channel_offset_ + v * 16 >= problem_size_.C, v);
    }
  }
  __forceinline__ __device__ void increment_filter()   {
    
    pointer_ += params_.inc_rs;
  }
  __forceinline__ __device__ void increment_filter(int num)   {
    
    pointer_ += params_.inc_rs * num;
  }
  __forceinline__ __device__ void reset_k()   {
    
    pointer_ += params_.inc_c_reset;
    reduce_channel_offset_ = reduce_channel_offset_backup_;
    mask_ = mask_backup_;
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
  __forceinline__ __device__ void add_byte_offset(int64_t byte_offset)   {
    
    pointer_ += byte_offset;
  }
  __forceinline__ __device__ void at()  const {
    
  }
  __forceinline__ __device__ bool valid(int s, int c, int ss, int v)  const {
    
    return mask_[v] & (1u << (s * 1 + c * 1 + ss));
  }
  __forceinline__ __device__ const tv::array<int8_t, 16> * get(int stride, int contig, int ss)  const {
    
    return reinterpret_cast<const tv::array<int8_t, 16> *>(pointer_ + contig * 32 * 8 / 8);
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<int8_t, 32, 0>& frag, int32_t pointer_offset)   {
    
    frag.clear();
    tv::array<int8_t, 16> *frag_ptr = reinterpret_cast<tv::array<int8_t, 16> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 2; ++s){
      TV_PRAGMA_UNROLL
      for (int c = 0; c < 1; ++c){
        TV_PRAGMA_UNROLL
        for (int ss = 0; ss < 1; ++ss){
          TV_PRAGMA_UNROLL
          for (int v = 0; v < 1; ++v){
            int idx = s * 1 + 
                c * 1 + ss * 1 + v;
            tv::array<int8_t, 16> const *access_ptr = get(s, c, ss) + v + pointer_offset / 16;
            // tv::gemm::global_load<tv::array<int8_t, 16>, sizeof(tv::array<int8_t, 16>)>(
            //     frag_ptr[idx], access_ptr, valid(s, c, ss));
            GlobalLoad::run(frag_ptr[idx], access_ptr, valid(s, c, ss, v));
          }
        }
      }
      if (s != 1){
          pointer_ += params_.inc_strided;
      }
    }
  }
  __forceinline__ __device__ const tv::array<int8_t, 16>* load_ptr_with_param(int s, int c, bool& valid_ref)   {
    
    tv::array<int8_t, 16> const *access_ptr = get(s, c, 0) + 0;
    valid_ref = valid(s, c, 0, 0);
    if (c == 0 && s != 1)
        pointer_ += params_.inc_strided;
    return access_ptr;
  }
  __forceinline__ __device__ void load_invalid()   {
    
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 2; ++s){
      if (s != 1){
          pointer_ += params_.inc_strided;
      }
    }
  }
  __forceinline__ __device__ void load(tv::array<int8_t, 32, 0>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void clear_mask()   {
    
  }
};
} // namespace inpiterb
} // namespace Turing_s8s8s8s32f16tnt_m32n64k32m32n32k32A1T8816_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib