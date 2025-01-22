#pragma once
#include <spconvlib/cumm/gemm/main/Turing_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T1688_200_S1/mma/mma_ns_wb/layout/MyTensorOpLayout.h>
#include <spconvlib/cumm/gemm/main/Turing_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T1688_200_S1/mma/mma_ns_wb/ldsm/LdMatrix.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T1688_200_S1 {
namespace mma {
namespace mma_ns_wb {
using TensorOpLayout = spconvlib::cumm::gemm::main::Turing_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T1688_200_S1::mma::mma_ns_wb::layout::MyTensorOpLayout;
using LdMatrix = spconvlib::cumm::gemm::main::Turing_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T1688_200_S1::mma::mma_ns_wb::ldsm::LdMatrix;
struct WarpIteratorCongruous {
  int wmma_k_index_;
  const tv::array<tv::half_t, 8> * pointer_[2];
  int32_t byte_offset_;
  __forceinline__ __device__  WarpIteratorCongruous(tv::half_t * ptr, int warp_idx_k, int warp_idx_mn, int lane_idx) : wmma_k_index_(0), byte_offset_(0)  {
    
    lane_idx %= 32;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
        int offset = TensorOpLayout::get_ldm_initial_offset<1, 4>(
            lane_idx, i, true);
        pointer_[i] = reinterpret_cast<const tv::array<tv::half_t, 8> * >(ptr + offset);
    }
    add_tile_offset(4 * warp_idx_k, warp_idx_mn);
  }
  __forceinline__ __device__ void add_pointer_offset(int64_t offset)   {
    byte_offset_ += offset * sizeof(tv::half_t);
  }
  __forceinline__ __device__ void add_tile_offset(int warp_idx_k, int warp_idx_mn, bool force_update = false)   {
    
        constexpr int kContigEqual = 64;
    int mn_offset = warp_idx_mn;
    int k_offset = warp_idx_k;
    if (64 < kContigEqual || force_update) {
      constexpr int kwarp_per_crosswise = 1;
      int warp_offset = warp_idx_mn & (kwarp_per_crosswise - 1);
      mn_offset = warp_idx_mn ^ warp_offset;            // kwarp_per_crosswise is 2^a
      warp_offset *= 2 / kwarp_per_crosswise;
      if (warp_offset || force_update) {
    const tv::array<tv::half_t, 8> * buffer[2];
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) 
      buffer[i] = pointer_[(i + warp_offset) % 2];
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) 
      pointer_[i] = buffer[i];
      }
    }
    int offset = (k_offset * 2048 +
                mn_offset * 64);
    add_pointer_offset(offset);
  }
  __forceinline__ __device__ void tile_increment(int num_tile)   {
    add_tile_offset(num_tile, 0);
  }
  __forceinline__ __device__ WarpIteratorCongruous & operator++()   {
    add_tile_offset(1, 0); // strided, contig
    // tv::printf2_block_once(threadIdx.x, "byte_offset_=", byte_offset_);
    if (1 > 1) {
        ++wmma_k_index_;
        // Jump to next stage
        if (wmma_k_index_ == 4) {
            wmma_k_index_ = 0;
            add_tile_offset(((1 - 1) * 4), 0);
        }
    }
    return *this;
  }
  __forceinline__ __device__ void load_with_byte_offset(tv::array<tv::half_t, 16, 0>& frag, int32_t byte_offset)   {
    tv::array<unsigned, 4> *fetch_ptr = 
    reinterpret_cast<tv::array<unsigned, 4> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 2; ++c) {
            int access_idx = c + s * 2;
            const tv::array<tv::half_t, 8> * source_ptr =
                pointer_[c % 2] +
                8 * (c / 2) +
                1 * s * 32;
            char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;
            LdMatrix::run(fetch_ptr[access_idx], source_byte_ptr);
        }
    }
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<tv::half_t, 16, 0>& frag, int32_t pointer_offset)   {
    load_with_byte_offset(frag, pointer_offset * sizeof(tv::half_t));
  }
  __forceinline__ __device__ void load(tv::array<tv::half_t, 16, 0>& frag)   {
    load_with_byte_offset(frag, 0);
  }
  __forceinline__ __device__ void set_kgroup_index(int wmma_k)   {
    
  }
};
} // namespace mma_ns_wb
} // namespace mma
} // namespace Turing_f16f16f16f16f16ttt_m128n256k32m64n64k32A1T1688_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib