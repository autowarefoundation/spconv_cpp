#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16ttt_m64n128k32m32n64k32A1T884_200_S1 {
namespace mma {
namespace mma_ns_wb {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
struct VoltaWarpTileIteratorCongruous {
  const tv::array<tv::half_t, 8> * pointers_[1];
  int32_t byte_offset_;
  __forceinline__ __device__  VoltaWarpTileIteratorCongruous(tv::half_t * ptr, int warp_idx_k, int warp_idx_mn, int lane_idx)   {
    
    int access_strided = (lane_idx >> 3) & 0x3;
    int access_contiguous = ((lane_idx ^ (lane_idx >> 3)) & 0x3);
    pointers_[0] = reinterpret_cast<const tv::array<tv::half_t, 8> *>(ptr) +
                    access_contiguous + access_strided * 16;
    add_warp_offset(warp_idx_k, warp_idx_mn);
  }
  __forceinline__ __device__ void add_warp_offset(int warp_idx_k, int warp_idx_mn)   {
    int mn_offset = warp_idx_mn;
    int k_offset = 8 * warp_idx_k;
    // TODO why?
    if (false) {
        if (64 == 32) {
            if (mn_offset % 2) {
                auto tmp_pointer = pointers_[0];
                pointers_[0] = pointers_[1];
                pointers_[1] = tmp_pointer;
            }
            mn_offset = mn_offset / 2 * 2;
        }
    }
    auto offset = k_offset * 16 * 4 *
                    8 +
                mn_offset * 64;
    // if (!Left){
    //   tv::printf2_block_once(threadIdx.x, offset);
    // }
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        pointers_[i] += offset / 8;
    }
  }
  __forceinline__ __device__ void tile_increment(int num)   {
    // this function is only called when move warp iter back to start offset.
    // so we need to reset wmma_k_index_
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        pointers_[i] += 16 * 4 * num;
    }
  }
  __forceinline__ __device__ VoltaWarpTileIteratorCongruous & operator++()   {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        pointers_[i] += 16 * 4;
    }
    return *this;
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<tv::half_t, 16, 0>& frag, int32_t pointer_offset)   {
    // pointer_offset: element unit
    int32_t byte_offset = pointer_offset * sizeof(tv::half_t);
    tv::array<tv::half_t, 8> * dst_ptr = reinterpret_cast<tv::array<tv::half_t, 8> *>(&frag);
    // kRow: 1, kCol: 2
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 2; ++c) {
            int idx = c + s * 2;
            const tv::array<tv::half_t, 8> * source_ptr;
            if (false) {
                source_ptr = pointers_[s & 1] + 32 * c +
                            4 * (s / 2) * 16;
            } else {
                source_ptr = pointers_[0] + 32 / 8 * c +
                            4 * s * 16;
            }
            char const *source_byte_ptr =
                reinterpret_cast<char const *>(source_ptr) + byte_offset +
                byte_offset_;
            // if (Left){
            //   auto  ppp = reinterpret_cast<const_pointer>(source_byte_ptr);
            //   tv::printf2_block_once(threadIdx.x, s, c, 
            //     reinterpret_cast<AccessType const*> (source_byte_ptr) - pointer_bkp_, 
            //     int(ppp[0]), int(ppp[1]), int(ppp[2]), int(ppp[3]));
            // }
            dst_ptr[idx] = *(reinterpret_cast<const tv::array<tv::half_t, 8> *>(source_byte_ptr));
        }
    }
  }
  __forceinline__ __device__ void load(tv::array<tv::half_t, 16, 0>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void set_kgroup_index(int wmma_k)   {
    
  }
};
} // namespace mma_ns_wb
} // namespace mma
} // namespace Volta_f16f16f16f16f16ttt_m64n128k32m32n64k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib