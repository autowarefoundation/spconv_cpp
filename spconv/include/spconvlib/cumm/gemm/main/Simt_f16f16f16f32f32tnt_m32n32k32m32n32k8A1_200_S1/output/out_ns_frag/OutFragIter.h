#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1 {
namespace output {
namespace out_ns_frag {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
struct OutFragIter {
  int index_;
  const tv::array<float, 4> * src_ptr_;
  __forceinline__ __device__  OutFragIter(const void* src_ptr) : src_ptr_(reinterpret_cast<const tv::array<float, 4> *>(src_ptr)), index_(0)  {
    
  }
  __forceinline__ __device__ void load(tv::array<float, 4, 0> & frag, int32_t index_offset = 0)   {
    tv::array<float, 4> * frag_ptr = reinterpret_cast<tv::array<float, 4> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int n = 0; n < 1; ++n) {
        frag_ptr[n] = src_ptr_[index_ * 1 + n];
    }
  }
  __forceinline__ __device__ OutFragIter& operator++()   {
    ++index_;
    return *this;
  }
};
} // namespace out_ns_frag
} // namespace output
} // namespace Simt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib