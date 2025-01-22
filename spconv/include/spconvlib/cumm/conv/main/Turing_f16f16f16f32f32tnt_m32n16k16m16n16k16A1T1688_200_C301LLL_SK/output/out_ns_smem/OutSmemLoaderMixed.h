#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A1T1688_200_C301LLL_SK/output/out_ns_smem/tmap/Out5DLinear.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A1T1688_200_C301LLL_SK {
namespace output {
namespace out_ns_smem {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::conv::main::Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A1T1688_200_C301LLL_SK::output::out_ns_smem::tmap::Out5DLinear;
struct OutSmemLoaderMixed {
  const tv::alignedarray<int4, 1, 16> * pointers_[1];
  __forceinline__ __device__  OutSmemLoaderMixed(float * ptr, int thread_idx)   {
    auto thread_offset = ThreadMap::initial_offset(thread_idx);
    auto pointer = reinterpret_cast<const tv::alignedarray<int4, 1, 16> *>(ptr);
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 1; ++i){
      int col_idx_in_subacc = (thread_offset[1] / 4) * 1;
      int smem_line_offset = (col_idx_in_subacc * 4 * 4 / 128) % 1;
      col_idx_in_subacc += (smem_line_offset + i) % 1;
      // tv::printf2_once<' ', -1>(i, threadIdx.x, "col_idx_in_subacc", thread_offset[0], thread_offset[1], col_idx_in_subacc, thread_offset[0] * 6 + col_idx_in_subacc);
      pointers_[i] = pointer + thread_offset[0] * 6 + col_idx_in_subacc;
    }
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<float, 4, 0> & frag, int32_t pointer_offset)   {
    tv::alignedarray<int4, 1, 16> *frag_ptr = reinterpret_cast<tv::alignedarray<int4, 1, 16> *>(&frag);
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster) {
        TV_PRAGMA_UNROLL
        for (int group = 0; group < 1; ++group) {
            TV_PRAGMA_UNROLL
            for (int row = 0; row < 1; ++row) {
                int row_ptr_offset = (
                    row * 48 +
                    group * 6 +
                    cluster * 6 +
                    pointer_offset / 4);
                int frag_row_idx = (row + 1 *
                                (group + 1 * cluster));
                TV_PRAGMA_UNROLL
                for (int column = 0; column < 1; ++column) {
                    int frag_idx = frag_row_idx * 1 + column;
                    int vector_idx = ((column * 16 /
                                    4) *
                                    1);
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < 1; ++v) {
                        auto mem_ptr = pointers_[v] + row_ptr_offset;
                        // tv::printf2_once<' ', -1>(cluster, group, row, column, v, frag_idx * 1 + v, vector_idx, mem_ptr - smem_pointer_);
                        frag_ptr[frag_idx * 1 +
                                 v] = (mem_ptr[vector_idx]);
                        // tv::print_ptr_once<int, 0, 4, -1>(reinterpret_cast<const int*>(mem_ptr));
                    }
                }
            }
        }
    }
  }
  __forceinline__ __device__ void load(tv::array<float, 4, 0> & frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    
    pointers_[0] += pointer_offset / 4;
  }
};
} // namespace out_ns_smem
} // namespace output
} // namespace Turing_f16f16f16f32f32tnt_m32n16k16m16n16k16A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib