#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f16tnt_m128n128k128m64n64k128A1T16832_200_C301LLL_SK_S8 {
namespace output {
namespace out_ns_frag {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
struct OutFragIterTensorOp {
  int index_;
  const tv::alignedarray<int2, 1, 8> * src_ptr_;
  __forceinline__ __device__  OutFragIterTensorOp(const void* src_ptr) : src_ptr_(reinterpret_cast<const tv::alignedarray<int2, 1, 8> *>(src_ptr)), index_(0)  {
    
  }
  __forceinline__ __device__ void load(tv::array<int32_t, 16, 0> & frag, int32_t index_offset = 0)   {
    tv::alignedarray<int2, 1, 8> * frag_ptr = reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(&frag);
    int index = index_ + index_offset;
    TV_PRAGMA_UNROLL
    for (int n = 0; n < 8; ++n) {
        int accumulator_access_offset = 
            index + n * 16 / 2;
        frag_ptr[n] = src_ptr_[accumulator_access_offset];
    }
  }
  __forceinline__ __device__ OutFragIterTensorOp& operator++()   {
    ++index_;
    return *this;
  }
};
} // namespace out_ns_frag
} // namespace output
} // namespace Ampere_s8s8s8s32f16tnt_m128n128k128m64n64k128A1T16832_200_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib