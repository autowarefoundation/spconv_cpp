#pragma once
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f32tnt_m128n64k32m64n32k32A1T16816_200_C301LLL_SKD_S8/mma/async_cp_iter_1_B/cp_async_copy/CpAsyncCopy.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f32tnt_m128n64k32m64n32k32A1T16816_200_C301LLL_SKD_S8 {
namespace mma {
namespace async_cp_iter_1_B {
using CpAsyncCp = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f32tnt_m128n64k32m64n32k32A1T16816_200_C301LLL_SKD_S8::mma::async_cp_iter_1_B::cp_async_copy::CpAsyncCopy;
struct AsyncCopyIteration {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {
    
    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 0, valid);
    CpAsyncCp::copy(dest_ptr, src_ptr, valid);
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {
    
    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 0, valid);
    CpAsyncCp::copy_zfill(dest_ptr, src_ptr, valid);
  }
};
} // namespace async_cp_iter_1_B
} // namespace mma
} // namespace Ampere_s8s8s8s32f32tnt_m128n64k32m64n32k32A1T16816_200_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib