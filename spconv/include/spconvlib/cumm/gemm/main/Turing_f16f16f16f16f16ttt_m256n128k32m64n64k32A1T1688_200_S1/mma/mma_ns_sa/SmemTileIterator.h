#pragma once
#include <spconvlib/cumm/gemm/main/Turing_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T1688_200_S1/inpitera/maskiter/PitchLinearWarpRaked.h>
#include <spconvlib/cumm/gemm/main/Turing_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T1688_200_S1/mma/mma_ns_wa/layout/MyTensorOpLayout.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T1688_200_S1 {
namespace mma {
namespace mma_ns_sa {
using ThreadMap = spconvlib::cumm::gemm::main::Turing_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T1688_200_S1::inpitera::maskiter::PitchLinearWarpRaked;
using Layout = spconvlib::cumm::gemm::main::Turing_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T1688_200_S1::mma::mma_ns_wa::layout::MyTensorOpLayout;
struct SmemTileIterator {
  tv::array<tv::half_t, 8> * pointer_[1];
  int32_t byte_offset_;
  __forceinline__ __device__  SmemTileIterator(int stride, tv::half_t * ptr, int thread_id) : byte_offset_(0)  {
    
    auto thread_offset_base = ThreadMap::initial_offset(thread_id);
    auto layout = Layout();
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        pointer_[i] = reinterpret_cast<tv::array<tv::half_t, 8> *>(
            ptr + layout(thread_offset_base[0] + i * 8,
                        thread_offset_base[1]));
    }
  }
  __forceinline__ __device__ tv::array<tv::half_t, 8> * get(int s, int c)  const {
    tv::array<tv::half_t, 8> * access_ptr = pointer_[s & 0];
    int external_stride_idx = (s & ~0);
    int access_offset = (external_stride_idx * 4 *
                     16 + c * 8);
    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);
    return reinterpret_cast<tv::array<tv::half_t, 8> *>(access_byte_ptr + byte_offset_);
  }
  __forceinline__ __device__ void add_pointer_offset(int64_t offset)   {
    byte_offset_ += offset * sizeof(tv::half_t);
  }
  __forceinline__ __device__ void add_tile_offset(int s, int c)   {
    add_pointer_offset(c * 64 +
        s * 16384);
  }
  __forceinline__ __device__ void tile_increment(int num_tile)   {
    add_tile_offset(0, num_tile);
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<tv::half_t, 32, 0> const& frag, int32_t pointer_offset)   {
    store_with_byte_offset(frag, pointer_offset * 16 / 8);
  }
  __forceinline__ __device__ void store_with_byte_offset(tv::array<tv::half_t, 32, 0> const& frag, int32_t byte_offset)   {
    
    const tv::array<tv::half_t, 8> * frag_ptr = reinterpret_cast<const tv::array<tv::half_t, 8> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 4; ++s) {
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 1; ++c) {
            int access_idx = c + s * 1;
            char *byte_ptr = reinterpret_cast<char *>(get(s, c)) + byte_offset;
            tv::array<tv::half_t, 8> * access_ptr = reinterpret_cast<tv::array<tv::half_t, 8> *>(byte_ptr);
            *access_ptr = frag_ptr[access_idx];
        }
    }
  }
  __forceinline__ __device__ tv::array<tv::half_t, 8> * store_ptr_with_param(int s, int c, bool& valid_ref)   {
    
    return reinterpret_cast<tv::array<tv::half_t, 8> *>(get(s, c));
  }
  __forceinline__ __device__ void store(tv::array<tv::half_t, 32, 0> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ SmemTileIterator & operator++()   {
    add_tile_offset(0, 1);
    return *this;
  }
};
} // namespace mma_ns_sa
} // namespace mma
} // namespace Turing_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T1688_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib