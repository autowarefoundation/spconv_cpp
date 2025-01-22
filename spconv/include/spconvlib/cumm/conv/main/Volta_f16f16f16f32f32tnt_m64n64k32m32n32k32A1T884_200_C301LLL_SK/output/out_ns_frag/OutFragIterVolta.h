#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Volta_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T884_200_C301LLL_SK {
namespace output {
namespace out_ns_frag {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
struct OutFragIterVolta {
  int index_;
  const tv::alignedarray<int2, 1, 8> * src_ptr_;
  __forceinline__ __device__  OutFragIterVolta(const void* src_ptr) : src_ptr_(reinterpret_cast<const tv::alignedarray<int2, 1, 8> *>(src_ptr)), index_(0)  {
    
  }
  __forceinline__ __device__ void load(tv::array<float, 16, 0> & frag, int32_t index_offset = 0)   {
    tv::alignedarray<int2, 1, 8> * frag_ptr = reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(&frag);
    constexpr int kRegsPerMmaRow = 2;
    TV_PRAGMA_UNROLL
    for (int reg_row = 0; reg_row < 2; ++reg_row) {
        TV_PRAGMA_UNROLL
        for (int tile_n = 0; tile_n < 1; ++tile_n) {
            TV_PRAGMA_UNROLL
            for (int mma_n = 0; mma_n < 2 * 2; ++mma_n) {
                // (index_ & 1): 01010101
                // (index_ & 0b10): 00110011
                // (index_ & 2) * Policy::MmaIterations::kCount / 2:
                // 00220022
                // (mma_n & 1) * 2: 02020202
                // (mma_n & 2) * 2: 00220022
                int mma_idx = (index_ & 1) +
                            (index_ & 2) * 4 / 2 +
                            (tile_n * 1) *
                                4 +
                            (mma_n & 1) * 2;
                int reg_offset = reg_row * kRegsPerMmaRow + (mma_n & 2) * 2;
                int reg_idx = mma_idx * 8 + reg_offset;
                *frag_ptr = src_ptr_[reg_idx / 2];
                ++frag_ptr;
            }
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
} // namespace Volta_f16f16f16f32f32tnt_m64n64k32m32n32k32A1T884_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib