#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/inpiterb/gload/GlobalLoad.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/inpiterb/mask/Mask.h>
#include <spconvlib/cumm/conv/main/Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/inpiterb/tmap/PitchLinearWarpRaked.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/cp/ConvProblem.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/iterb_p/WeightOptParams.h>
#include <spconvlib/cumm/conv/main/cpTuring_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8/lb/TensorGeneric.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8 {
namespace inpiterb {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using GlobalLoad = spconvlib::cumm::conv::main::Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::inpiterb::gload::GlobalLoad;
using Mask = spconvlib::cumm::conv::main::Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::inpiterb::mask::Mask;
using ThreadMap = spconvlib::cumm::conv::main::Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::inpiterb::tmap::PitchLinearWarpRaked;
using ConvProblem = spconvlib::cumm::conv::main::cpTuring_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::cp::ConvProblem;
using Params = spconvlib::cumm::conv::main::cpTuring_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::iterb_p::WeightOptParams;
using Layout = spconvlib::cumm::conv::main::cpTuring_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8::lb::TensorGeneric;
struct WeightIteratorDP4AV2Mask {
  Params const& params_;
  ConvProblem const& problem_size_;
  const char * pointer_;
  int reduce_channel_offset_;
  int reduce_channel_offset_backup_;
  Mask mask_backup_;
  Mask mask_;
  __forceinline__ __device__  WeightIteratorDP4AV2Mask(Params const& params, ConvProblem const& problem_size, const int8_t * ptr, int thread_id, const tv::array<int, 2>& threadblock_offset) : params_(params), problem_size_(problem_size), pointer_(reinterpret_cast<const char *>(ptr))  {
    
    auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
    mask_.clear();
    reduce_channel_offset_ = thread_offset[1];
    reduce_channel_offset_backup_ = thread_offset[1];
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 0 < problem_size.C), 0, 0, 0, 0);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 1 < problem_size.C), 0, 0, 0, 1);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 2 < problem_size.C), 0, 0, 0, 2);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 3 < problem_size.C), 0, 0, 0, 3);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 4 < problem_size.C), 0, 0, 0, 4);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 5 < problem_size.C), 0, 0, 0, 5);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 6 < problem_size.C), 0, 0, 0, 6);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 7 < problem_size.C), 0, 0, 0, 7);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 8 < problem_size.C), 0, 0, 0, 8);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 9 < problem_size.C), 0, 0, 0, 9);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 10 < problem_size.C), 0, 0, 0, 10);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 11 < problem_size.C), 0, 0, 0, 11);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 12 < problem_size.C), 0, 0, 0, 12);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 13 < problem_size.C), 0, 0, 0, 13);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 14 < problem_size.C), 0, 0, 0, 14);
    mask_.set_coord((thread_offset[0] + 0 < problem_size.K) && (thread_offset[1] + 15 < problem_size.C), 0, 0, 0, 15);
    pointer_ += (thread_offset[0] * params.layout.strides[0] + thread_offset[1]) * 8 / 8;
    mask_backup_ = mask_;
  }
  __forceinline__ __device__ void operator++()   {
    
  }
  __forceinline__ __device__ void increment_k()   {
    
    pointer_ += params_.inc_c;
    reduce_channel_offset_ += params_.filter_c_delta;
    mask_.mask_[0] = reduce_channel_offset_ + 0 >= problem_size_.C ? 
        mask_.mask_[0] & 4294967294u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 1 >= problem_size_.C ? 
        mask_.mask_[0] & 4294967293u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 2 >= problem_size_.C ? 
        mask_.mask_[0] & 4294967291u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 3 >= problem_size_.C ? 
        mask_.mask_[0] & 4294967287u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 4 >= problem_size_.C ? 
        mask_.mask_[0] & 4294967279u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 5 >= problem_size_.C ? 
        mask_.mask_[0] & 4294967263u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 6 >= problem_size_.C ? 
        mask_.mask_[0] & 4294967231u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 7 >= problem_size_.C ? 
        mask_.mask_[0] & 4294967167u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 8 >= problem_size_.C ? 
        mask_.mask_[0] & 4294967039u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 9 >= problem_size_.C ? 
        mask_.mask_[0] & 4294966783u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 10 >= problem_size_.C ? 
        mask_.mask_[0] & 4294966271u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 11 >= problem_size_.C ? 
        mask_.mask_[0] & 4294965247u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 12 >= problem_size_.C ? 
        mask_.mask_[0] & 4294963199u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 13 >= problem_size_.C ? 
        mask_.mask_[0] & 4294959103u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 14 >= problem_size_.C ? 
        mask_.mask_[0] & 4294950911u : mask_.mask_[0];
    mask_.mask_[0] = reduce_channel_offset_ + 15 >= problem_size_.C ? 
        mask_.mask_[0] & 4294934527u : mask_.mask_[0];
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
  __forceinline__ __device__ void clear_all_mask_if_not_pred(bool pred)   {
    
    return mask_.clear_all_mask_if_not_pred(pred);
  }
  __forceinline__ __device__ void clear_all_mask_if_pred(bool pred)   {
    
    return mask_.clear_all_mask_if_pred(pred);
  }
  __forceinline__ __device__ void add_byte_offset(int64_t byte_offset)   {
    
    pointer_ += byte_offset;
  }
  __forceinline__ __device__ void at()  const {
    
  }
  __forceinline__ __device__ bool valid(int s, int c, int ss, int v)  const {
    
    return mask_.query_coord(s, c, ss, v);
    // return mask_[v] & (1u << (s * 1 + c * 1 + ss));
  }
  __forceinline__ __device__ const tv::array<int8_t, 1> * get(int stride, int contig, int ss)  const {
    
    return reinterpret_cast<const tv::array<int8_t, 1> *>(pointer_ + contig * 32 * 8 / 8);
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
          tv::array<int8_t, 1> const *access_ptr = get(s, c, ss) + pointer_offset / 1;
          TV_PRAGMA_UNROLL
          for (int v = 0; v < 16; ++v){
            int idx = s * 16 + 
                c * 16 + ss * 16 + v;
            // tv::array<int8_t, 1> const *access_ptr = get(s, c, ss) + v + pointer_offset / 1;
            // tv::gemm::global_load<tv::array<int8_t, 1>, sizeof(tv::array<int8_t, 1>)>(
            //     frag_ptr[idx], access_ptr, valid(s, c, ss));
            GlobalLoad::run(frag_ptr[idx], access_ptr + v, valid(s, c, ss, v));
          }
        }
      }
      if (s != 0){
          pointer_ += params_.inc_strided;
      }
    }
  }
  __forceinline__ __device__ void load_invalid()   {
    
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s){
      if (s != 0){
          pointer_ += params_.inc_strided;
      }
    }
  }
  __forceinline__ __device__ void load(tv::array<int8_t, 16, 0>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void clear_mask()   {
    
  }
};
} // namespace inpiterb
} // namespace Turing_s8s8s8s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib