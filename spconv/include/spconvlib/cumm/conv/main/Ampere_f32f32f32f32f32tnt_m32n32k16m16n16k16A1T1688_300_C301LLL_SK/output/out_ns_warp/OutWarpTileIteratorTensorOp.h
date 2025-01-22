#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/layout/RowMajor.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK {
namespace output {
namespace out_ns_warp {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using RowMajor = spconvlib::cumm::gemm::layout::RowMajor;
struct OutWarpTileIteratorTensorOp {
  tv::alignedarray<int2, 1, 8> * pointer_;
  RowMajor layout_;
  __forceinline__ __device__  OutWarpTileIteratorTensorOp(float * ptr, int warp_offset_m, int warp_offset_n, int lane_idx) : pointer_(reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(ptr)), layout_(20)  {
    
    // pointer_bkp_ = reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(ptr);
    int quad_id = (lane_idx / 4); 
    int lane_in_quad = (lane_idx % 4);
    pointer_ += layout_(quad_id, lane_in_quad);
    add_warp_offset(warp_offset_m, warp_offset_n);
    // tv::printf2_block_once(threadIdx.x, pointer_ - reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(ptr));
  }
  __forceinline__ __device__ void add_warp_offset(int warp_m, int warp_n)   {
    pointer_ += layout_(warp_m * 8, warp_n * 
        16 / 2);
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<float, 4, 0> const & frag, int32_t pointer_offset)   {
    const tv::alignedarray<int2, 1, 8> * frag_ptr = reinterpret_cast<const tv::alignedarray<int2, 1, 8> *>(&frag);
    // tv::printf2_block_once(threadIdx.x, pointer_ - pointer_bkp_);
    TV_PRAGMA_UNROLL
    for (int n = 0; n < 2; ++n) {
        pointer_[n * 4 + pointer_offset / 2] = frag_ptr[n];
    }
  }
  __forceinline__ __device__ void store(tv::array<float, 4, 0> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    pointer_ += pointer_offset / 2;
  }
};
} // namespace out_ns_warp
} // namespace output
} // namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib