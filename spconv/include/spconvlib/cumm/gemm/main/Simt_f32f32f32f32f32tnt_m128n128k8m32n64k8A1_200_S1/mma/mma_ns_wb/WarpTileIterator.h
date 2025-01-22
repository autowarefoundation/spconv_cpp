#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32tnt_m128n128k8m32n64k8A1_200_S1/mma/mma_ns_wb/ns1/RowMajorInterleaved.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32tnt_m128n128k8m32n64k8A1_200_S1/mma/mma_ns_wa/ns2/RowMajorInterleaved.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m128n128k8m32n64k8A1_200_S1 {
namespace mma {
namespace mma_ns_wb {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using SmemLayout = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32tnt_m128n128k8m32n64k8A1_200_S1::mma::mma_ns_wb::ns1::RowMajorInterleaved;
using LaneLayout = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32tnt_m128n128k8m32n64k8A1_200_S1::mma::mma_ns_wa::ns2::RowMajorInterleaved;
struct WarpTileIterator {
  tv::array<float, 4> * pointer_;
  __forceinline__ __device__  WarpTileIterator(float * ptr, int warp_idx_k, int warp_idx_residual, int lane_idx)   {
    constexpr auto lane_layout = LaneLayout::from_shape({4, 8});
    constexpr auto smem_layout = SmemLayout::from_shape({8, 132});
    tv::array<int, 2> logical_offset{warp_idx_k * 8,
                                    warp_idx_residual * 64};
    // int lane_offset;
    if (false) {
        // 0, 1, 0, 1, 0, 1, ..., 2, 3, 2, 3, 2, 3
        logical_offset[1] += lane_layout.inverse_0(lane_idx) * 4;
    } else {
        // 0, 0, 1, 1, 2, 2, 3, 3, ...
        logical_offset[1] += lane_layout.inverse_1(lane_idx) * 4;
    }
    auto offset = smem_layout(logical_offset[0], logical_offset[1]);
    pointer_ = reinterpret_cast<tv::array<float, 4> *>(ptr + offset);
  }
  __forceinline__ __device__ void tile_increment(int num)   {
    static constexpr auto smem_layout = SmemLayout::from_shape({8, 132});
    pointer_ += smem_layout(num * 1, 0) / 4;
  }
  __forceinline__ __device__ WarpTileIterator & operator++()   {
    static constexpr auto smem_layout = SmemLayout::from_shape({8, 132});
    auto offset = 
    pointer_ += smem_layout(1, 0) / 4;
    return *this;
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<float, 8, 0>& frag, int32_t pointer_offset)   {
    static constexpr auto smem_layout = SmemLayout::from_shape({8, 132});
    tv::array<float, 4> * dst_ptr = reinterpret_cast<tv::array<float, 4> *>(&frag);
    // kRow: 1, kCol: 2
    TV_PRAGMA_UNROLL
    for (int k = 0; k < 1; ++k) {
        TV_PRAGMA_UNROLL
        for (int n = 0; n < 2; ++n) {
            // ref offset: [8, 128 / 4], (0, n * 8) ~= (0, 32)
            // lane_id = 0, [0, 0-3], [0, 32-35]
            // lane_id = 1, [0, 0-3], [0, 32-35]
            // lane_id = 2, [0, 4-7], [0, 36-39]
            // lane_id = 3, [0, 4-7], [0, 36-39]
            // ...
            // lane_id = 31, [0, 28-31], [0, 60-63]
            auto offset = smem_layout(k * 1,
                                    n * (false ? 4 : 8) *
                                        4) /
                        4;
            // auto offset = smem_layout(k * 1,
            //                         n * 8);
            TV_PRAGMA_UNROLL
            for (int sub = 0; sub < 1; ++sub){
                dst_ptr[k * 2 
                    + n * 1 + sub] =
                    pointer_[offset + sub + pointer_offset / 4];
            }
        }
    }
  }
  __forceinline__ __device__ void load(tv::array<float, 8, 0>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void set_kgroup_index(int wmma_k)   {
    
  }
};
} // namespace mma_ns_wb
} // namespace mma
} // namespace Simt_f32f32f32f32f32tnt_m128n128k8m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib