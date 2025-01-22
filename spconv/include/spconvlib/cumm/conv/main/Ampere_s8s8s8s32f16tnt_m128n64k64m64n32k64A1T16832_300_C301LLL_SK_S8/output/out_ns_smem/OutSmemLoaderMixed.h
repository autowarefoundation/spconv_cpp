#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n64k64m64n32k64A1T16832_300_C301LLL_SK_S8/output/out_ns_smem/tmap/Out5DLinear.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f16tnt_m128n64k64m64n32k64A1T16832_300_C301LLL_SK_S8 {
namespace output {
namespace out_ns_smem {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n64k64m64n32k64A1T16832_300_C301LLL_SK_S8::output::out_ns_smem::tmap::Out5DLinear;
struct OutSmemLoaderMixed {
  const tv::alignedarray<int4, 1, 16> * pointers_[2];
  __forceinline__ __device__  OutSmemLoaderMixed(int32_t * ptr, int thread_idx)   {
    auto thread_offset = ThreadMap::initial_offset(thread_idx);
    auto pointer = reinterpret_cast<const tv::alignedarray<int4, 1, 16> *>(ptr);
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i){
      int lane_col_idx = thread_offset[1] / 8;
      int lane_offset = ((lane_col_idx % 8) * 2) | ((lane_col_idx / 4) ^ i);
      pointers_[i] = pointer + thread_offset[0] * 18 + lane_offset;
    }
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<int32_t, 8, 0> & frag, int32_t pointer_offset)   {
    tv::alignedarray<int4, 1, 16> *frag_ptr = reinterpret_cast<tv::alignedarray<int4, 1, 16> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster) {
        TV_PRAGMA_UNROLL
        for (int group = 0; group < 1; ++group) {
            TV_PRAGMA_UNROLL
            for (int row = 0; row < 1; ++row) {
                int row_ptr_offset = (
                    row * 72 +
                    group * 18 +
                    cluster * 18 +
                    pointer_offset / 4);
                int frag_row_idx = (row + 1 *
                                (group + 1 * cluster));
                TV_PRAGMA_UNROLL
                for (int column = 0; column < 1; ++column) {
                    int frag_idx = frag_row_idx * 1 + column;
                    int vector_idx = ((column * 64 /
                                    8) *
                                    2);
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < 2; ++v) {
                        auto mem_ptr = pointers_[v] + row_ptr_offset;
                        // tv::printf2_once<' ', -1>(cluster, group, row, column, v, frag_idx * 2 + v, vector_idx, mem_ptr - smem_pointer_);
                        frag_ptr[frag_idx * 2 +
                                 v] = (mem_ptr[vector_idx]);
                        // tv::print_ptr_once<int, 0, 4, -1>(reinterpret_cast<const int*>(mem_ptr));
                    }
                }
            }
        }
    }
  }
  __forceinline__ __device__ void load(tv::array<int32_t, 8, 0> & frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    
    pointers_[0] += pointer_offset / 4;
    pointers_[1] += pointer_offset / 4;
  }
};
} // namespace out_ns_smem
} // namespace output
} // namespace Ampere_s8s8s8s32f16tnt_m128n64k64m64n32k64A1T16832_300_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib