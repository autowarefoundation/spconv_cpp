#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16tnt_m128n256k32m64n64k32A1T884_200_S1 {
namespace mma {
namespace mma_ns_wa {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
struct VoltaWarpTileIteratorCrosswise {
  const tv::array<tv::half_t, 8> * pointer_;
  int32_t byte_offset_;
  int32_t wmma_k_index_;
  __forceinline__ __device__  VoltaWarpTileIteratorCrosswise(tv::half_t * ptr, int warp_idx_k, int warp_idx_mn, int lane_idx) : wmma_k_index_(0)  {
    
    int quad = (lane_idx / 4);
    int lane_in_quad = (lane_idx % 4);
    int access_contiguous;
    // swizzle id: tid[4]|tid[1:0]|(tid[2]^tid[4])
    access_contiguous = ((quad & 0x4) << 1) + ((lane_in_quad) << 1) +
                        ((quad & 0x1) ^ ((quad & 0x4) >> 2));
    byte_offset_ = access_contiguous * sizeof(tv::half_t) * 8;
    pointer_ = reinterpret_cast<const tv::array<tv::half_t, 8> *>(ptr);
    add_warp_offset(warp_idx_k, warp_idx_mn);
  }
  __forceinline__ __device__ void add_warp_offset(int warp_idx_k, int warp_idx_mn)   {
    int mn_offset = warp_idx_mn;
    int k_offset = 8 * warp_idx_k;
    // kTileShapeKM: [K, M|N]
    // TODO better offset
    auto offset = k_offset * 4 * 16 +
                mn_offset * 64 * 4 / 8;
    // printf2_block_once(threadIdx.x, offset);
    pointer_ += offset;
  }
  __forceinline__ __device__ void tile_increment(int num)   {
    // this function is only called when move warp iter back to start offset.
    // so we need to reset wmma_k_index_
    wmma_k_index_ = 0;
    // tv::printf2_block_once("tile_increment_warp", threadIdx.x, kLineSize * num, kLineSize, num);
    pointer_ += 64 * num;
  }
  __forceinline__ __device__ VoltaWarpTileIteratorCrosswise & operator++()   {
    wmma_k_index_ = (wmma_k_index_ + 1) & 7;
    // handle permute (i)
    if (wmma_k_index_ == 4 || wmma_k_index_ == 0) {
        // ptr swapped in k = 4-7, so we 'swap' ptr here.
        // byte_offset_ -=(+=) self.sizeof_element * self.kElementsPerAccess
        byte_offset_ ^= 1 * sizeof(tv::half_t) * 8;
    }
    pointer_ += 64;
    return *this;
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<tv::half_t, 16, 0>& frag, int32_t pointer_offset)   {
    int32_t byte_offset = pointer_offset * sizeof(tv::half_t);
    tv::array<tv::half_t, 8> * dst_ptr = reinterpret_cast<tv::array<tv::half_t, 8> *>(&frag);
    // kRow: 1, kCol: 2
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 2; ++s) {
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 1; ++c) {
            int idx = c + s * 1;
            const tv::array<tv::half_t, 8> * source_ptr =
                pointer_ + 1 * c * 64 + 32 * s / 2;
            char const *source_byte_ptr =
                reinterpret_cast<char const *>(source_ptr) + byte_offset +
                byte_offset_;
            dst_ptr[idx] = *(reinterpret_cast<const tv::array<tv::half_t, 8> *>(source_byte_ptr));
            if (wmma_k_index_ & 0x2) {
                uint64_t *low = reinterpret_cast<uint64_t *>(&frag) + idx * 2;
                uint64_t *high = reinterpret_cast<uint64_t *>(&frag) + idx * 2 + 1;
                uint64_t tmp = *low;
                *low = *high;
                *high = tmp;
            }
        }
    }
  }
  __forceinline__ __device__ void load(tv::array<tv::half_t, 16, 0>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void set_kgroup_index(int wmma_k)   {
    wmma_k_index_ = wmma_k;
  }
};
} // namespace mma_ns_wa
} // namespace mma
} // namespace Volta_f16f16f16f16f16tnt_m128n256k32m64n64k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib