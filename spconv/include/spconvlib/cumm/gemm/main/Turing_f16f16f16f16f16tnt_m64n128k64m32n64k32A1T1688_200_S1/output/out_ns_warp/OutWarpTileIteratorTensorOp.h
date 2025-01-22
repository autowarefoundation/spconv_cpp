#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/layout/RowMajor.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_f16f16f16f16f16tnt_m64n128k64m32n64k32A1T1688_200_S1 {
namespace output {
namespace out_ns_warp {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using RowMajor = spconvlib::cumm::gemm::layout::RowMajor;
struct OutWarpTileIteratorTensorOp {
  tv::alignedarray<int, 1, 4> * pointer_;
  RowMajor layout_;
  __forceinline__ __device__  OutWarpTileIteratorTensorOp(tv::half_t * ptr, int warp_offset_m, int warp_offset_n, int lane_idx) : pointer_(reinterpret_cast<tv::alignedarray<int, 1, 4> *>(ptr)), layout_(68)  {
    
    // pointer_bkp_ = reinterpret_cast<tv::alignedarray<int, 1, 4> *>(ptr);
    int quad_id = (lane_idx / 4); 
    int lane_in_quad = (lane_idx % 4);
    pointer_ += layout_(quad_id, lane_in_quad);
    add_warp_offset(warp_offset_m, warp_offset_n);
    // tv::printf2_block_once(threadIdx.x, pointer_ - reinterpret_cast<tv::alignedarray<int, 1, 4> *>(ptr));
  }
  __forceinline__ __device__ void add_warp_offset(int warp_m, int warp_n)   {
    pointer_ += layout_(warp_m * 8, warp_n * 
        64 / 2);
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<tv::half_t, 16, 0> const & frag, int32_t pointer_offset)   {
    const tv::alignedarray<int, 1, 4> * frag_ptr = reinterpret_cast<const tv::alignedarray<int, 1, 4> *>(&frag);
    // tv::printf2_block_once(threadIdx.x, pointer_ - pointer_bkp_);
    TV_PRAGMA_UNROLL
    for (int n = 0; n < 8; ++n) {
        pointer_[n * 4 + pointer_offset / 2] = frag_ptr[n];
    }
  }
  __forceinline__ __device__ void store(tv::array<tv::half_t, 16, 0> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    pointer_ += pointer_offset / 2;
  }
};
} // namespace out_ns_warp
} // namespace output
} // namespace Turing_f16f16f16f16f16tnt_m64n128k64m32n64k32A1T1688_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib