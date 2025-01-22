#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1/inpitera/maskiter/PitchLinear.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1 {
namespace mma {
namespace mma_ns_sa {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1::inpitera::maskiter::PitchLinear;
struct SmemTileIteratorV2 {
  char * pointer_;
  __forceinline__ __device__  SmemTileIteratorV2(int stride, tv::half_t* ptr, int thread_id)   {
    auto thread_offset = ThreadMap::initial_offset(thread_id);
    int offset = (thread_offset[1] / 1) * 136
         + thread_offset[0] * 1;
    pointer_ = reinterpret_cast<char *>(ptr + offset);
    // for transposed input, kThreadAccessShape and kIterationDelta is
    // transposed too.
    // inc_strided_ = stride * 1 * sizeof(tv::half_t);
    // if (false) {
    //     inc_advance_ = 128 * sizeof(tv::half_t);
    // } else {
    //     inc_advance_ = 8 * stride * sizeof(tv::half_t);
    // }
    // tv::printf2_block_once(threadIdx.x, "inc_strided_", inc_strided_, inc_advance_, stride_, offset);
  }
  __forceinline__ __device__ void tile_increment(int num)   {
    pointer_ += 2176 * num;
  }
  __forceinline__ __device__ SmemTileIteratorV2& operator++()   {
    pointer_ +=  2176;
    return *this;
  }
  __forceinline__ __device__ SmemTileIteratorV2& operator--()   {
    pointer_ -=  2176;
    return *this;
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<tv::half_t, 4, 0>& frag, int32_t pointer_offset)   {
    tv::alignedarray<tv::half_t, 1, 2> *frag_ptr = reinterpret_cast<tv::alignedarray<tv::half_t, 1, 2> *>(&frag);
    const char * byte_pointer =
        pointer_ + pointer_offset * sizeof(tv::half_t);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        tv::alignedarray<tv::half_t, 1, 2> const *access_ptr =
            reinterpret_cast<tv::alignedarray<tv::half_t, 1, 2> const *>(byte_pointer);
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 4; ++c) {
            int idx = c + s * 4;
            frag_ptr[idx] =
                access_ptr[c * 32 / 1];
        }
        if (s < 1 - 1) {
            byte_pointer +=  272;
        }
    }
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<tv::half_t, 4, 0> const & frag, int32_t pointer_offset)   {
    tv::alignedarray<tv::half_t, 1, 2> const *frag_ptr = reinterpret_cast<tv::alignedarray<tv::half_t, 1, 2> const *>(&frag);
    char * byte_pointer =
        pointer_ + pointer_offset * sizeof(tv::half_t);
    TV_PRAGMA_UNROLL
    for (int s = 0; s < 1; ++s) {
        tv::alignedarray<tv::half_t, 1, 2> *access_ptr =
            reinterpret_cast<tv::alignedarray<tv::half_t, 1, 2> *>(byte_pointer);
        TV_PRAGMA_UNROLL
        for (int c = 0; c < 4; ++c) {
            int idx = c + s * 4;
            access_ptr[c * 32 / 1] =
                frag_ptr[idx];
        }
        if (s < 0) {
            byte_pointer += 272;
        }
    }
  }
  __forceinline__ __device__ void store(tv::array<tv::half_t, 4, 0> const & frag)   {
    store_with_pointer_offset(frag, 0);
  }
};
} // namespace mma_ns_sa
} // namespace mma
} // namespace Simt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib