#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/layout/RowMajor.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f32tnt_m64n32k32m32n32k32A1T16816_200_C301LLL_SK_S8 {
namespace output {
namespace out_ns_warp {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using RowMajor = spconvlib::cumm::gemm::layout::RowMajor;
struct OutWarpTileIteratorTensorOp {
  tv::alignedarray<int2, 1, 8> * pointer_;
  RowMajor layout_;
  __forceinline__ __device__  OutWarpTileIteratorTensorOp(int32_t * ptr, int warp_offset_m, int warp_offset_n, int lane_idx) : pointer_(reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(ptr)), layout_(20)  {
    
    // pointer_bkp_ = reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(ptr);
    int quad_id = (lane_idx / 4); 
    int lane_in_quad = (lane_idx % 4);
    pointer_ += layout_(quad_id, lane_in_quad);
    add_warp_offset(warp_offset_m, warp_offset_n);
    // tv::printf2_block_once(threadIdx.x, pointer_ - reinterpret_cast<tv::alignedarray<int2, 1, 8> *>(ptr));
  }
  __forceinline__ __device__ void add_warp_offset(int warp_m, int warp_n)   {
    pointer_ += layout_(warp_m * 8, warp_n * 
        32 / 2);
  }
  __forceinline__ __device__ void store_with_pointer_offset(tv::array<int32_t, 8, 0> const & frag, int32_t pointer_offset)   {
    const tv::alignedarray<int2, 1, 8> * frag_ptr = reinterpret_cast<const tv::alignedarray<int2, 1, 8> *>(&frag);
    // tv::printf2_block_once(threadIdx.x, pointer_ - pointer_bkp_);
    TV_PRAGMA_UNROLL
    for (int n = 0; n < 4; ++n) {
        pointer_[n * 4 + pointer_offset / 2] = frag_ptr[n];
    }
  }
  __forceinline__ __device__ void store(tv::array<int32_t, 8, 0> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    pointer_ += pointer_offset / 2;
  }
};
} // namespace out_ns_warp
} // namespace output
} // namespace Ampere_s8s8s8s32f32tnt_m64n32k32m32n32k32A1T16816_200_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib