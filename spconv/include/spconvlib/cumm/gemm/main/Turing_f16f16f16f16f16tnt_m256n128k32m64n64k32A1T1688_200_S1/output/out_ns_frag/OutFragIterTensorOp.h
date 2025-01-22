#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_f16f16f16f16f16tnt_m256n128k32m64n64k32A1T1688_200_S1 {
namespace output {
namespace out_ns_frag {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
struct OutFragIterTensorOp {
  int index_;
  const tv::alignedarray<int, 1, 4> * src_ptr_;
  __forceinline__ __device__  OutFragIterTensorOp(const void* src_ptr) : src_ptr_(reinterpret_cast<const tv::alignedarray<int, 1, 4> *>(src_ptr)), index_(0)  {
    
  }
  __forceinline__ __device__ void load(tv::array<tv::half_t, 16, 0> & frag, int32_t index_offset = 0)   {
    tv::alignedarray<int, 1, 4> * frag_ptr = reinterpret_cast<tv::alignedarray<int, 1, 4> *>(&frag);
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
} // namespace Turing_f16f16f16f16f16tnt_m256n128k32m64n64k32A1T1688_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib