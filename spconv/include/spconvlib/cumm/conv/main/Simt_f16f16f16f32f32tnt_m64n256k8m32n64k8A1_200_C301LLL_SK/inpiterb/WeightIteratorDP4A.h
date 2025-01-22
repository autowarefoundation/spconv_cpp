#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/Simt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK/inpiterb/gload/GlobalLoad.h>
#include <spconvlib/cumm/conv/main/Simt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK/inpiterb/tmap/PitchLinear.h>
#include <spconvlib/cumm/conv/main/cpSimt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK/cp/ConvProblem.h>
#include <spconvlib/cumm/conv/main/cpSimt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK/iterb_p/WeightOptParams.h>
#include <spconvlib/cumm/conv/main/cpSimt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK/lb/TensorGeneric.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK {
namespace inpiterb {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using GlobalLoad = spconvlib::cumm::conv::main::Simt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK::inpiterb::gload::GlobalLoad;
using ThreadMap = spconvlib::cumm::conv::main::Simt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK::inpiterb::tmap::PitchLinear;
using ConvProblem = spconvlib::cumm::conv::main::cpSimt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK::cp::ConvProblem;
using Params = spconvlib::cumm::conv::main::cpSimt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK::iterb_p::WeightOptParams;
using Layout = spconvlib::cumm::conv::main::cpSimt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK::lb::TensorGeneric;
struct WeightIteratorDP4A {
  Params const& params_;
  ConvProblem const& problem_size_;
  const char * pointer_;
  int reduce_channel_offset_;
  int reduce_channel_offset_backup_;
  tv::array<uint32_t, 1> mask_backup_;
  tv::array<uint32_t, 1> mask_;
  __forceinline__ __device__  WeightIteratorDP4A(Params const& params, ConvProblem const& problem_size, const tv::half_t * ptr, int thread_id, const tv::array<int, 2>& threadblock_offset) : params_(params), problem_size_(problem_size), pointer_(reinterpret_cast<const char *>(ptr))  {
    
    auto thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id);
    mask_.clear();
    reduce_channel_offset_ = thread_offset[1];
    reduce_channel_offset_backup_ = thread_offset[1];
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 8; ++s){
      TV_PRAGMA_UNROLL
      for (int c = 0; c < 1; ++c){
        TV_PRAGMA_UNROLL
        for (int ss = 0; ss < 1; ++ss){
          uint32_t pred = thread_offset[0] + s * 32 + ss < problem_size.K;
          TV_PRAGMA_UNROLL
          for (int v = 0; v < 1; ++v){
            mask_[v] |= (pred << (s * 1 + c * 1 + ss));
          }
        }
      }
    }
    TV_PRAGMA_UNROLL
    for (int v = 0; v < 1; ++v){
        mask_[v] = thread_offset[1] + v * 1 >= problem_size.C ? 0 : mask_[v];
    }
    pointer_ += (thread_offset[0] * params.layout.strides[0] + thread_offset[1]) * 16 / 8;
    mask_backup_ = mask_;
  }
  __forceinline__ __device__ void operator++()   {
    
  }
  __forceinline__ __device__ void increment_k()   {
    
    pointer_ += params_.inc_c;
    reduce_channel_offset_ += params_.filter_c_delta;
    TV_PRAGMA_UNROLL
    for (int v = 0; v < 1; ++v){
        clear_mask_if_pred(reduce_channel_offset_ + v * 1 >= problem_size_.C, v);
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
  __forceinline__ __device__ const tv::array<tv::half_t, 1> * get(int stride, int contig, int ss)  const {
    
    return reinterpret_cast<const tv::array<tv::half_t, 1> *>(pointer_ + contig * 1 * 16 / 8);
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<tv::half_t, 8, 0>& frag, int32_t pointer_offset)   {
    
    frag.clear();
    tv::array<tv::half_t, 1> *frag_ptr = reinterpret_cast<tv::array<tv::half_t, 1> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 8; ++s){
      TV_PRAGMA_UNROLL
      for (int c = 0; c < 1; ++c){
        TV_PRAGMA_UNROLL
        for (int ss = 0; ss < 1; ++ss){
          TV_PRAGMA_UNROLL
          for (int v = 0; v < 1; ++v){
            int idx = s * 1 + 
                c * 1 + ss * 1 + v;
            tv::array<tv::half_t, 1> const *access_ptr = get(s, c, ss) + v + pointer_offset / 1;
            // tv::gemm::global_load<tv::array<tv::half_t, 1>, sizeof(tv::array<tv::half_t, 1>)>(
            //     frag_ptr[idx], access_ptr, valid(s, c, ss));
            GlobalLoad::run(frag_ptr[idx], access_ptr, valid(s, c, ss, v));
          }
        }
      }
      if (s != 7){
          pointer_ += params_.inc_strided;
      }
    }
  }
  __forceinline__ __device__ const tv::array<tv::half_t, 1>* load_ptr_with_param(int s, int c, bool& valid_ref)   {
    
    tv::array<tv::half_t, 1> const *access_ptr = get(s, c, 0) + 0;
    valid_ref = valid(s, c, 0, 0);
    if (c == 0 && s != 7)
        pointer_ += params_.inc_strided;
    return access_ptr;
  }
  __forceinline__ __device__ void load_invalid()   {
    
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 8; ++s){
      if (s != 7){
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
} // namespace Simt_f16f16f16f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib