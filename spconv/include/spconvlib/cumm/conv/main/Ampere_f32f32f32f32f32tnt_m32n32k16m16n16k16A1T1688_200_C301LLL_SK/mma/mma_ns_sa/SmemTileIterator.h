#pragma once
#include <spconvlib/cumm/conv/main/Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK/inpitera/tmap/PitchLinearWarpRaked.h>
#include <spconvlib/cumm/conv/main/Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK/mma/mma_ns_wa/layout/MyTensorOpLayout.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK {
namespace mma {
namespace mma_ns_sa {
using ThreadMap = spconvlib::cumm::conv::main::Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK::inpitera::tmap::PitchLinearWarpRaked;
using Layout = spconvlib::cumm::conv::main::Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK::mma::mma_ns_wa::layout::MyTensorOpLayout;
struct SmemTileIterator {
  tv::array<float, 4> * pointer_[1];
  int32_t byte_offset_;
  __forceinline__ __device__  SmemTileIterator(int stride, float * ptr, int thread_id) : byte_offset_(0)  {
    
    auto thread_offset_base = ThreadMap::initial_offset(thread_id);
    auto layout = Layout();
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        pointer_[i] = reinterpret_cast<tv::array<float, 4> *>(
            ptr + layout(thread_offset_base[0] + i * 8,
                        thread_offset_base[1]));
    }
  }
  __forceinline__ __device__ tv::array<float, 4> * get(int s, int c)  const {
    tv::array<float, 4> * access_ptr = pointer_[s & 0];
    int external_stride_idx = (s & ~0);
    int access_offset = (external_stride_idx * 4 *
                     16 + c * 8);
    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);
    return reinterpret_cast<tv::array<float, 4> *>(access_byte_ptr + byte_offset_);
  }
  __forceinline__ __device__ void add_pointer_offset(int64_t offset)   {
    byte_offset_ += offset * sizeof(float);
  }
  __forceinline__ __device__ void add_tile_offset(int s, int c)   {
    add_pointer_offset(c * 32 +
        s * 1024);
  }
  __forceinline__ __device__ void tile_increment(int num_tile)   {
    add_tile_offset(0, num_tile);
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<float, 4, 0> const& frag, int32_t pointer_offset)   {
    store_with_byte_offset(frag, pointer_offset * 32 / 8);
  }
  __forceinline__ __device__ void store_with_byte_offset(tv::array<float, 4, 0> const& frag, int32_t byte_offset)   {
    
    const tv::array<float, 4> * frag_ptr = reinterpret_cast<const tv::array<float, 4> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 1; ++c) {
            int access_idx = c + s * 1;
            char *byte_ptr = reinterpret_cast<char *>(get(s, c)) + byte_offset;
            tv::array<float, 4> * access_ptr = reinterpret_cast<tv::array<float, 4> *>(byte_ptr);
            *access_ptr = frag_ptr[access_idx];
        }
    }
  }
  __forceinline__ __device__ tv::array<float, 4> * store_ptr_with_param(int s, int c, bool& valid_ref)   {
    
    return reinterpret_cast<tv::array<float, 4> *>(get(s, c));
  }
  __forceinline__ __device__ void store(tv::array<float, 4, 0> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ SmemTileIterator & operator++()   {
    add_tile_offset(0, 1);
    return *this;
  }
};
} // namespace mma_ns_sa
} // namespace mma
} // namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib