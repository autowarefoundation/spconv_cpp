#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/Volta_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T884_200_S1/inpiterb/maskiter/PitchLinearWarpRaked.h>
#include <spconvlib/cumm/gemm/main/gpVolta_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T884_200_S1/iterb_p/MaskTileIteratorParams.h>
#include <spconvlib/cumm/gemm/main/Volta_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T884_200_S1/inpiterb/maskiter/GlobalLoad.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T884_200_S1 {
namespace inpiterb {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T884_200_S1::inpiterb::maskiter::PitchLinearWarpRaked;
using Params = spconvlib::cumm::gemm::main::gpVolta_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T884_200_S1::iterb_p::MaskTileIteratorParams;
using GlobalLoad = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T884_200_S1::inpiterb::maskiter::GlobalLoad;
struct MaskTileIterator {
  const char * pointer_;
  Params const & params_;
  tv::array<int, 2> extent_;
  tv::array<int, 2> thread_offset_;
  int residue_offset_;
  bool is_residue_tile_;
  uint32_t predicates_[1];
  __forceinline__ __device__  MaskTileIterator(Params const & params, const tv::half_t * ptr, tv::array<int, 2> extent, int thread_id, const tv::array<int, 2>& threadblock_offset) : params_(params), pointer_(reinterpret_cast<const char *>(ptr)), extent_(extent), is_residue_tile_(true)  {
    
    int residue_size = (extent[0] - threadblock_offset[0]) %
                    32;
    if (!residue_size) {
        residue_size = 32;
    }
    // tv::printf2_once(params_.inc_strided_, params_.inc_next_);
    residue_offset_ = residue_size;
        tv::array<int, 2> residue_extent{
    std::min(threadblock_offset[0] + residue_size, extent_[0])
    , extent_[1]};
    // residue tile always first k axis tile
    // thread_id / kAccessShape[kContig] is 'sub-tile' coord, so we need to
    // convert back to element coord
    thread_offset_ = threadblock_offset + ThreadMap::initial_offset(thread_id);
    // auto init = ThreadMap::initial_offset(thread_id);
    if TV_IF_CONSTEXPR (!true) {
        thread_offset_[0] += residue_offset_;
    }
    // tv::printf2_block_once(threadIdx.x, thread_offset_[0] * extent_[1] + thread_offset_[1]);
    if TV_IF_CONSTEXPR (!true) {
        compute_predicates_(extent, false);
    } else {
        compute_predicates_(residue_extent, false);
    }
    // here we can't use extent_[1] because splitk may split stride.
    add_pointer_offset(thread_offset_[0] * params.stride_ + thread_offset_[1]);
  }
  __forceinline__ __device__ void update_indices()   {
    
  }
  __forceinline__ __device__ void add_pointer_offset(int64_t offset)   {
    pointer_ += sizeof(tv::half_t) * offset;
  }
  __forceinline__ __device__ void tile_increment(int num_tile)   {
    
    if (is_residue_tile_){
      thread_offset_[0] += residue_offset_;
      pointer_ += sizeof(tv::half_t) * params_.stride_ * residue_offset_;
      compute_predicates_(extent_, true);
      pointer_ += params_.inc_advance_ * (num_tile - 1);
    }
    else{
      pointer_ += params_.inc_advance_ * num_tile;
    }
    is_residue_tile_ = false;
  }
  __forceinline__ __device__ const tv::alignedarray<int4, 1, 16> * get(int s, int c, int v)  const {
    
    return reinterpret_cast<const tv::alignedarray<int4, 1, 16> *>(
            pointer_ + 
            (c * 64) *
                sizeof(tv::half_t)) +
        v;
  }
  __forceinline__ __device__ void inc_stride()   {
    pointer_ += params_.inc_strided_; 
  }
  __forceinline__ __device__ void end_iter()   {
    pointer_ += params_.inc_next_;
    pointer_ -= params_.inc_advance_;
  }
  __forceinline__ __device__ bool valid(int s, int c, int v)   {
    
    int scalar_index =
        s * 4 + c * 1 +
        v * 1;
    int word_idx = scalar_index / 16;
    int residual = scalar_index % 16;
    int byte_idx = residual / 4;
    int bit_idx = residual % 4;
    bool pred = (predicates_[word_idx] & (1u << (byte_idx * 8 + bit_idx))) != 0;
    return pred;
  }
  __forceinline__ __device__ void clear_mask()   {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        predicates_[i] = 0u;
    }
  }
  __forceinline__ __device__ MaskTileIterator & operator++()   {
    tile_increment(1);
    return *this;
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<tv::half_t, 32, 0>& frag, int32_t pointer_offset)   {
    load_with_byte_offset(frag, pointer_offset * sizeof(tv::half_t));
  }
  __forceinline__ __device__ void store_with_pointer_offset()   {
    
  }
  __forceinline__ __device__ const tv::alignedarray<int4, 1, 16>* load_ptr_with_param(int s, int c, bool& valid_ref)   {
    
    auto ret_ptr = get(s, c, 0);
    valid_ref = valid(s, c, 0);
    if(c == 3){
        if(s != 0)
            inc_stride();
        else
            end_iter();
        }
    return ret_ptr;
  }
  __forceinline__ __device__ void load_with_byte_offset(tv::array<tv::half_t, 32, 0>& frag, int64_t byte_offset)   {
    
    tv::alignedarray<int4, 1, 16> *frag_ptr = reinterpret_cast<tv::alignedarray<int4, 1, 16>  *>(&frag);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s){
      TV_PRAGMA_UNROLL
      for (int c = 0; c < 4; ++c){
        TV_PRAGMA_UNROLL
        for (int v = 0; v < 1; ++v){
          int idx =
              s * 4 + c * 1 +
              v * 1;
          char const *byte_ptr =
              reinterpret_cast<char const *>(get(s, c, v)) + byte_offset;
          tv::alignedarray<int4, 1, 16> const *access_ptr =
              reinterpret_cast<tv::alignedarray<int4, 1, 16> const *>(byte_ptr);
          GlobalLoad::run(frag_ptr[idx], access_ptr, valid(s, c, v));
          // tv::gemm::global_load<tv::alignedarray<int4, 1, 16>, sizeof(tv::alignedarray<int4, 1, 16>)>(
          //    frag_ptr[idx], access_ptr, valid(s, c, v));
        }
      }
      if (s != 1 - 1) {
          inc_stride();
      }
    }
    end_iter();
  }
  __forceinline__ __device__ void store_with_byte_offset()   {
    
  }
  __forceinline__ __device__ void load(tv::array<tv::half_t, 32, 0>& frag)   {
    load_with_byte_offset(frag, 0);
  }
  __forceinline__ __device__ void store()   {
    
  }
  __forceinline__ __device__ void compute_predicates_(tv::array<int, 2> extent, bool steady = false)   {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        predicates_[i] = 0;
    }
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 4; ++c) {
            TV_PRAGMA_UNROLL
            for (int v = 0; v < 1; ++v) {
                tv::array<int, 2> elem_coord{
                    s * 4 + thread_offset_[0],
                    c * 64 + v * 8 +
                        thread_offset_[1]};
                bool valid;
                if (steady) {
                    if (0 == 1) {
                        valid = elem_coord[0] < extent[0];
                    } else {
                        valid = elem_coord[1] < extent[1];
                    }
                } else {
                    valid = elem_coord[0] < extent[0] &&
                            elem_coord[1] < extent[1];
                }
                int scalar_index =
                    s * 4 + c * 1 +
                    v * 1;
                int word_idx = scalar_index / 16;
                int residual = scalar_index % 16;
                int byte_idx = residual / 4;
                int bit_idx = residual % 4;
                predicates_[word_idx] |=
                    (unsigned(valid) << (byte_idx * 8 + bit_idx));
            }
        }
    }
  }
};
} // namespace inpiterb
} // namespace Volta_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib