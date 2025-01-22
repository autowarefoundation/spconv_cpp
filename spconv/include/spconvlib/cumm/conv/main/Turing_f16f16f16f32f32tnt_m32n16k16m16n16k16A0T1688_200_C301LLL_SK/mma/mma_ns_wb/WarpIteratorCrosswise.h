#pragma once
#include <spconvlib/cumm/conv/main/Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK/mma/mma_ns_wb/layout/MyTensorOpLayout.h>
#include <spconvlib/cumm/conv/main/Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK/mma/mma_ns_wb/ldsm/LdMatrix.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK {
namespace mma {
namespace mma_ns_wb {
using TensorOpLayout = spconvlib::cumm::conv::main::Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK::mma::mma_ns_wb::layout::MyTensorOpLayout;
using LdMatrix = spconvlib::cumm::conv::main::Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK::mma::mma_ns_wb::ldsm::LdMatrix;
struct WarpIteratorCrosswise {
  const tv::array<tv::half_t, 8> * pointer_;
  int32_t byte_offset_;
  int wmma_k_index_;
  __forceinline__ __device__  WarpIteratorCrosswise(tv::half_t * ptr, int warp_idx_k, int warp_idx_mn, int lane_idx) : pointer_(reinterpret_cast<const tv::array<tv::half_t, 8> *>(ptr)), wmma_k_index_(0), byte_offset_(0)  {
    
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750))
        lane_idx = lane_idx % (2 * 8);
    #endif
    int offset_e = TensorOpLayout::get_ldm_initial_offset<2, 1>(
        lane_idx, 0, true);
    byte_offset_ = offset_e * 16 / 8;
    add_tile_offset(2 * warp_idx_k, warp_idx_mn);
  }
  __forceinline__ __device__ void add_tile_offset(int warp_idx_k, int warp_idx_mn)   {
    int mn_offset = warp_idx_mn;
    int k_offset = warp_idx_k;
    int sw_part_idx = k_offset / 2;
    int idx_in_sw_part = k_offset % 2;
    byte_offset_ ^= (idx_in_sw_part * 16);
    // tv::printf2_block_once(threadIdx.x, "premuteK", byte_offset_);
    pointer_ +=
        mn_offset * 64 +
        sw_part_idx * 8;
  }
  __forceinline__ __device__ void tile_increment(int num_tile)   {
    add_tile_offset(num_tile, 0);
  }
  __forceinline__ __device__ WarpIteratorCrosswise & operator++()   {
    
    if (((wmma_k_index_ & 0) & 1) == 0){
        // bit 0 advance
        byte_offset_ ^= 0b1 * 16;
    }
    else if ((wmma_k_index_ & 0) == 0b1){
        // bit 1 advance
        byte_offset_ ^= 0b11 * 16;
    }
    else if ((wmma_k_index_ & 0) == 0b11){
        // bit 2 advance
        byte_offset_ ^= 0b111 * 16;
    }
    wmma_k_index_++;
    if (wmma_k_index_ == 2) {
        wmma_k_index_ = 0;
        // k group increment
        add_tile_offset(2, 0);
    }
    return *this;
  }
  __forceinline__ __device__ void load_with_byte_offset(tv::array<tv::half_t, 4, 0>& frag, int32_t byte_offset)   {
    tv::array<unsigned, 2> *fetch_ptr =
        reinterpret_cast<tv::array<unsigned, 2> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 1; ++c) {
            int access_idx = c + s * 1;
            const tv::array<tv::half_t, 8> * source_ptr =
                pointer_ + 1 * c +
                8 * 2 * s * 
                4;
            char const *source_byte_ptr =
                reinterpret_cast<char const *>(source_ptr) + byte_offset +
                byte_offset_;
            LdMatrix::run(fetch_ptr[access_idx], source_byte_ptr);
        }
    }
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<tv::half_t, 4, 0>& frag, int32_t pointer_offset)   {
    load_with_byte_offset(frag, pointer_offset * sizeof(tv::half_t));
  }
  __forceinline__ __device__ void load(tv::array<tv::half_t, 4, 0>& frag)   {
    load_with_byte_offset(frag, 0);
  }
  __forceinline__ __device__ void set_kgroup_index(int wmma_k)   {
    wmma_k_index_ = wmma_k % (2);
  }
};
} // namespace mma_ns_wb
} // namespace mma
} // namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A0T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib