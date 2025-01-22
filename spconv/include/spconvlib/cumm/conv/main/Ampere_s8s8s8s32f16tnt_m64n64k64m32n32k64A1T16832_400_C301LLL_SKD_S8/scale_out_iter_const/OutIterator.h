#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f16tnt_m64n64k64m32n32k64A1T16832_400_C301LLL_SKD_S8/out_params_ns/tmap/Out5DLinear.h>
#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f16tnt_m64n64k64m32n32k64A1T16832_400_C301LLL_SKD_S8/out_params_scalebias_ns/OutIteratorParams.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f16tnt_m64n64k64m32n32k64A1T16832_400_C301LLL_SKD_S8 {
namespace scale_out_iter_const {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f16tnt_m64n64k64m32n32k64A1T16832_400_C301LLL_SKD_S8::out_params_ns::tmap::Out5DLinear;
using Params = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f16tnt_m64n64k64m32n32k64A1T16832_400_C301LLL_SKD_S8::out_params_scalebias_ns::OutIteratorParams;
struct OutIterator {
  const tv::half_t * pointer_;
  Params const& params_;
  bool column_masks_[1][2];
  __forceinline__ __device__  OutIterator(Params const& params, const tv::half_t * ptr, tv::array<int, 2> extent, tv::array<int, 2> offset_2d, int thread_idx) : params_(params)  {
    // pointer_bkp_ = ptr;
    auto thread_offset = ThreadMap::initial_offset(thread_idx) + offset_2d;
    TV_PRAGMA_UNROLL
    for (int c = 0; c < 1; ++c) {
        for (int v = 0; v < 2; ++v){
            column_masks_[c][v] = ((thread_offset[1] + 64 * c + v * 4) < extent[1]);
        }
    }
    // tv::printf2_block_once("Outthread_offset ", threadIdx.x, thread_offset[0], thread_offset[1], (thread_offset[0]) * stride + thread_offset[1]);
    pointer_ = ptr + thread_offset[1];
  }
  __forceinline__ __device__ void store_with_offset(tv::array<tv::half_t, 8, 0> const & frag, int32_t offset)   {
    
  }
  __forceinline__ __device__ void load_with_offset(tv::array<tv::half_t, 8, 0>  & frag, int32_t offset)   {
    
    auto cur_pointer = pointer_;
    tv::alignedarray<int2, 1, 8>  *frag_ptr = reinterpret_cast<tv::alignedarray<int2, 1, 8>  *>(&frag);
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster){
      TV_PRAGMA_UNROLL
      for (int group = 0; group < 1; ++group){
        TV_PRAGMA_UNROLL
        for (int row = 0; row < 1; ++row){
          int frag_row_idx =
              (row +  1 * (group +  1 * cluster));
          // delta: [Cluster, Group, Row]
          tv::half_t const *memory_pointer = cur_pointer + offset;
          TV_PRAGMA_UNROLL
          for (int column = 0; column < 1; ++column){
            TV_PRAGMA_UNROLL
            for (int v = 0; v < 2; ++v){
              bool guard = column_masks_[column][v];
              tv::gemm::global_load<tv::alignedarray<int2, 1, 8>, sizeof(tv::alignedarray<int2, 1, 8>)>(
                  frag_ptr[frag_row_idx *  2 + column * 2 + v],
                  memory_pointer + column * 64 + v * 4,
                  guard);
            }
          }
        }
      }
    }
  }
  __forceinline__ __device__ void load(tv::array<tv::half_t, 8, 0> & frag)   {
    load_with_offset(frag, 0);
  }
  __forceinline__ __device__ OutIterator& operator++()   {
    
    return *this;
  }
};
} // namespace scale_out_iter_const
} // namespace Ampere_s8s8s8s32f16tnt_m64n64k64m32n32k64A1T16832_400_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib