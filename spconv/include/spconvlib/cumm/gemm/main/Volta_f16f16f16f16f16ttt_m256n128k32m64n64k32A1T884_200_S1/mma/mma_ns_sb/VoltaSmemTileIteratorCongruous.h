#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/Volta_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T884_200_S1/inpiterb/maskiter/PitchLinearWarpRaked.h>
#include <spconvlib/cumm/gemm/main/Volta_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T884_200_S1/mma/mma_ns_sb/layout/VoltaTensorOpCongruous.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T884_200_S1 {
namespace mma {
namespace mma_ns_sb {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T884_200_S1::inpiterb::maskiter::PitchLinearWarpRaked;
using Layout = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T884_200_S1::mma::mma_ns_sb::layout::VoltaTensorOpCongruous;
struct VoltaSmemTileIteratorCongruous {
  tv::array<tv::half_t, 8> * pointers_[1];
  __forceinline__ __device__  VoltaSmemTileIteratorCongruous(int stride, tv::half_t * ptr, int thread_id)   {
    auto layout = Layout(stride);
    auto thread_offset_base = ThreadMap::initial_offset(thread_id);
    // int offs[2];
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        pointers_[i] = reinterpret_cast<tv::array<tv::half_t, 8> *>(
            ptr + layout(thread_offset_base[0] + i * 4,
                        thread_offset_base[1]));
    }
  }
  __forceinline__ __device__ void tile_increment(int num_tile)   {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        pointers_[i] +=
            num_tile * 4096 / 8;
    }
  }
  __forceinline__ __device__ VoltaSmemTileIteratorCongruous & operator++()   {
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i) {
        pointers_[i] += 4096 / 8;
    }
    return *this;
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<tv::half_t, 16, 0> const& frag, int32_t pointer_offset)   {
    const tv::array<tv::half_t, 8> * frag_ptr = reinterpret_cast<const tv::array<tv::half_t, 8> *>(&frag);
    int32_t vec_pointer_offset = pointer_offset / 8;
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        // TODO remove this
        tv::array<tv::half_t, 8> * access_ptr = pointers_[s & 1];
        // check next tile
        int stride_idx = (s & ~1);
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 2; ++c) {
            int idx = c + s * 2;
            int access_offset =
                stride_idx * 4 * 128 / 8 +
                c * 64 / 8 +
                vec_pointer_offset;
            // tv::printf2_block_once(threadIdx.x, s, c, access_offset);
            access_ptr[access_offset] = frag_ptr[idx];
        }
    }
  }
  __forceinline__ __device__ void store(tv::array<tv::half_t, 16, 0> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
};
} // namespace mma_ns_sb
} // namespace mma
} // namespace Volta_f16f16f16f16f16ttt_m256n128k32m64n64k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib