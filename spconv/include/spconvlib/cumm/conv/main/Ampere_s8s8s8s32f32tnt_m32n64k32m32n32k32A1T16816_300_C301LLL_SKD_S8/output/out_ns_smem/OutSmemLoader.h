#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f32tnt_m32n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8/output/out_ns_smem/tmap/Out5DLinear.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f32tnt_m32n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8 {
namespace output {
namespace out_ns_smem {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using ThreadMap = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f32tnt_m32n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8::output::out_ns_smem::tmap::Out5DLinear;
struct OutSmemLoader {
  int32_t * pointer_;
  __forceinline__ __device__  OutSmemLoader(int32_t * ptr, int thread_idx)   {
    auto thread_offset = ThreadMap::initial_offset(thread_idx);
    pointer_ = ptr + thread_offset[0] * 72 + thread_offset[1];
  }
  __forceinline__ __device__ void load_with_pointer_offset(tv::array<int32_t, 8, 0> & frag, int32_t pointer_offset)   {
    TV_PRAGMA_UNROLL
    for (int cluster = 0; cluster < 1; ++cluster) {
        TV_PRAGMA_UNROLL
        for (int group = 0; group < 1; ++group) {
            TV_PRAGMA_UNROLL
            for (int row = 0; row < 1; ++row) {
                const int32_t * cur_pointer =
                    pointer_ + row * 4 * 72 + group * 1 * 72 +
                    cluster * 1 * 72 + pointer_offset;
                int frag_row_idx =
                    (row + 1 * (group + 1 * cluster));
                tv::alignedarray<int4, 1, 16> *frag_ptr = reinterpret_cast<tv::alignedarray<int4, 1, 16> *>(&frag);
                tv::alignedarray<int4, 1, 16> const *memory_pointer =
                    reinterpret_cast<tv::alignedarray<int4, 1, 16> const *>(cur_pointer);
                TV_PRAGMA_UNROLL
                for (int column = 0; column < 1; ++column) {
                    int frag_idx = frag_row_idx * 1 + column;
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < 2; ++v) {
                        frag_ptr[frag_idx * 2 + v] =
                            memory_pointer[(column * 64 / 8) *
                                                2 +
                                            v];
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
    pointer_ += pointer_offset;
  }
};
} // namespace out_ns_smem
} // namespace output
} // namespace Ampere_s8s8s8s32f32tnt_m32n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib