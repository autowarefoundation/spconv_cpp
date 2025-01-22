#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK/inpiterb/gload/GlobalLoad.h>
#include <spconvlib/cumm/conv/main/Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK/inpiterb/mask/Mask.h>
#include <spconvlib/cumm/conv/main/Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK/inpiterb/tmap/PitchLinearWarpRaked.h>
#include <spconvlib/cumm/conv/main/cpTuring_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK/cp/ConvProblem.h>
#include <spconvlib/cumm/conv/main/cpTuring_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK/iterb_p/WeightOptParams.h>
#include <spconvlib/cumm/conv/main/cpTuring_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK/lb/TensorGeneric.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK {
namespace inpiterb {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using GlobalLoad = spconvlib::cumm::conv::main::Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK::inpiterb::gload::GlobalLoad;
using Mask = spconvlib::cumm::conv::main::Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK::inpiterb::mask::Mask;
using ThreadMap = spconvlib::cumm::conv::main::Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK::inpiterb::tmap::PitchLinearWarpRaked;
using ConvProblem = spconvlib::cumm::conv::main::cpTuring_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK::cp::ConvProblem;
using Params = spconvlib::cumm::conv::main::cpTuring_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK::iterb_p::WeightOptParams;
using Layout = spconvlib::cumm::conv::main::cpTuring_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK::lb::TensorGeneric;
struct WeightIteratorDP4AV2Mask {
  Params const& params_;
  ConvProblem const& problem_size_;
  const char * pointer_;
  int reduce_channel_offset_;
  int reduce_channel_offset_backup_;
  Mask mask_backup_;
  Mask mask_;
  __forceinline__ __device__  WeightIteratorDP4AV2Mask(Params const& params, ConvProblem const& problem_size, const tv::half_t * ptr, int thread_id, const tv::array<int, 2>& threadblock_offset) : params_(params), problem_size_(problem_size), pointer_(reinterpret_cast<const char *>(ptr))  {
    
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
    pointer_ += (thread_offset[0] * params.layout.strides[0] + thread_offset[1]) * 16 / 8;
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
  __forceinline__ __device__ const tv::array<tv::half_t, 1> * get(int stride, int contig, int ss)  const {
    
    return reinterpret_cast<const tv::array<tv::half_t, 1> *>(pointer_ + contig * 16 * 16 / 8);
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<tv::half_t, 8, 0>& frag, int32_t pointer_offset)   {
    
    frag.clear();
    tv::array<tv::half_t, 1> *frag_ptr = reinterpret_cast<tv::array<tv::half_t, 1> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s){
      TV_PRAGMA_UNROLL
      for (int c = 0; c < 1; ++c){
        TV_PRAGMA_UNROLL
        for (int ss = 0; ss < 1; ++ss){
          tv::array<tv::half_t, 1> const *access_ptr = get(s, c, ss) + pointer_offset / 1;
          TV_PRAGMA_UNROLL
          for (int v = 0; v < 8; ++v){
            int idx = s * 8 + 
                c * 8 + ss * 8 + v;
            // tv::array<tv::half_t, 1> const *access_ptr = get(s, c, ss) + v + pointer_offset / 1;
            // tv::gemm::global_load<tv::array<tv::half_t, 1>, sizeof(tv::array<tv::half_t, 1>)>(
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
  __forceinline__ __device__ void load(tv::array<tv::half_t, 8, 0>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void clear_mask()   {
    
  }
};
} // namespace inpiterb
} // namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib