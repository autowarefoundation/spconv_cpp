#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/out_params_ns/tmap/Out5DLinear.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/out_params_ns/OutIteratorParams.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1 {
namespace out_iter_const {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::out_params_ns::tmap::Out5DLinear;
using Params = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::out_params_ns::OutIteratorParams;
struct OutIterator {
  const tv::half_t * pointer_;
  Params const& params_;
  bool column_masks_[4][1];
  int32_t extent_row_;
  int32_t thread_start_row_;
  int counts_[3];
  int64_t indices_[2];
  __forceinline__ __device__  OutIterator(Params const& params, const tv::half_t * ptr, tv::array<int, 2> extent, tv::array<int, 2> offset_2d, int thread_idx) : params_(params)  {
    // pointer_bkp_ = ptr;
    counts_[0] = 0;
    counts_[1] = 0;
    counts_[2] = 0;
    auto thread_offset = ThreadMap::initial_offset(thread_idx) + offset_2d;
    TV_PRAGMA_UNROLL
    for (int c = 0; c < 4; ++c) {
        for (int v = 0; v < 1; ++v){
            column_masks_[c][v] = ((thread_offset[1] + 32 * c + v * 1) < extent[1]);
        }
    }
    // tv::printf2_block_once("Outthread_offset ", threadIdx.x, thread_offset[0], thread_offset[1], (thread_offset[0]) * stride + thread_offset[1]);
    extent_row_ = extent[0];
    thread_start_row_ = thread_offset[0];
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster){
      TV_PRAGMA_UNROLL
      for (int group = 0; group < 2; ++group){
        TV_PRAGMA_UNROLL
        for (int row = 0; row < 1; ++row){
          int idx = (row +  1 * (group +  2 * cluster));
          int row_offset =
              row * 1 + group * 16 + cluster * 1;
          bool row_guard = ((row_offset + thread_start_row_) < extent_row_ && params_.indice_ptr_ != nullptr);
          indices_[idx] = row_guard ? int64_t(params_.indice_ptr_[row_offset + thread_start_row_]) * int64_t(params_.stride) : 0;
        }
      }
    }
    pointer_ = ptr + thread_offset[1];
  }
  __forceinline__ __device__ void store_with_offset(tv::array<tv::half_t, 8, 0> const & frag, int32_t offset)   {
    
  }
  __forceinline__ __device__ void load_with_offset(tv::array<tv::half_t, 8, 0>  & frag, int32_t offset)   {
    
    auto cur_pointer = pointer_;
    tv::alignedarray<tv::half_t, 1, 2>  *frag_ptr = reinterpret_cast<tv::alignedarray<tv::half_t, 1, 2>  *>(&frag);
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster){
      TV_PRAGMA_UNROLL
      for (int group = 0; group < 2; ++group){
        TV_PRAGMA_UNROLL
        for (int row = 0; row < 1; ++row){
          int frag_row_idx =
              (row +  1 * (group +  2 * cluster));
          // delta: [Cluster, Group, Row]
          int row_offset =
              row * 1 + group * 16 + cluster * 1;
          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);
          tv::alignedarray<tv::half_t, 1, 2> const *memory_pointer =
              reinterpret_cast<tv::alignedarray<tv::half_t, 1, 2> const *>(cur_pointer + offset + indices_[frag_row_idx]);
          TV_PRAGMA_UNROLL
          for (int column = 0; column < 4; ++column){
            bool guard = row_guard && column_masks_[column][0];
            tv::gemm::global_load<tv::alignedarray<tv::half_t, 1, 2>, sizeof(tv::alignedarray<tv::half_t, 1, 2>)>(
                frag_ptr[frag_row_idx *  4 + column],
                (const void *)&memory_pointer[column * 32 / 1],
                guard);
          }
        }
      }
    }
  }
  __forceinline__ __device__ void store(tv::array<tv::half_t, 8, 0> const & frag)   {
    store_with_offset(frag, 0);
  }
  __forceinline__ __device__ void load(tv::array<tv::half_t, 8, 0> & frag)   {
    load_with_offset(frag, 0);
  }
  __forceinline__ __device__ OutIterator& operator++()   {
    ++counts_[2];
    // kPartShape: [Tile, Cluster, Group, Row, Col]
    thread_start_row_ += 1;
    if (counts_[2] == 8) {
    counts_[2] = 0;
    ++counts_[1];
    thread_start_row_ +=
        (4 - 1) * 1 * 8;
    if (counts_[1] == 1) {
        counts_[1] = 0;
        ++counts_[0];
        thread_start_row_ +=
            1 * 4 * 1 * 8;
        if (counts_[0] == 1) {
        counts_[0] = 0;
        }
    }
    }
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster){
      TV_PRAGMA_UNROLL
      for (int group = 0; group < 2; ++group){
        TV_PRAGMA_UNROLL
        for (int row = 0; row < 1; ++row){
          int idx =
              (row +  1 * (group +  2 * cluster));
          int row_offset =
              row * 1 + group * 16 + cluster * 1;
          bool row_guard = ((row_offset + thread_start_row_) < extent_row_ && params_.indice_ptr_ != nullptr);
          indices_[idx] = row_guard ? int64_t(params_.indice_ptr_[row_offset + thread_start_row_]) * int64_t(params_.stride) : 0;
        }
      }
    }
    return *this;
  }
};
} // namespace out_iter_const
} // namespace Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib