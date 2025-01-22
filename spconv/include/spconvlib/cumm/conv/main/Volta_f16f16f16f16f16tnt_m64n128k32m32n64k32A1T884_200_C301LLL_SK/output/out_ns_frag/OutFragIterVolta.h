#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_C301LLL_SK {
namespace output {
namespace out_ns_frag {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
struct OutFragIterVolta {
  int index_;
  const tv::alignedarray<int2, 1, 8> * src_ptr_;
  __forceinline__ __device__  OutFragIterVolta(const void* src_ptr) : src_ptr_(reinterpret_cast<const tv::alignedarray<int2, 1, 8> *>(src_ptr)), index_(0)  {
    
  }
  __forceinline__ __device__ void load(tv::array<tv::half_t, 32, 0> & frag, int32_t index_offset = 0)   {
    tv::alignedarray<int2, 1, 8> * frag_ptr = reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(&frag);
    static constexpr int kAccessesPerMma =
        8 / 4;
    TV_PRAGMA_UNROLL
    for (int tile_n = 0; tile_n < 2; ++tile_n) {
        int tile_access_idx =
            (tile_n * 1 + (index_ & 2) / 2) *
            2 * 2 * kAccessesPerMma;
        TV_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < 2 * kAccessesPerMma;
            ++mma_n) {
            int mma_access_idx =
                ((mma_n & 1) * 2 + (index_ & 1)) * kAccessesPerMma +
                (mma_n & 2) / 2;
            frag_ptr[tile_n * 2 * kAccessesPerMma + mma_n] =
                src_ptr_[tile_access_idx + mma_access_idx];
        }
    }
  }
  __forceinline__ __device__ OutFragIterVolta& operator++()   {
    ++index_;
    return *this;
  }
};
} // namespace out_ns_frag
} // namespace output
} // namespace Volta_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T884_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib