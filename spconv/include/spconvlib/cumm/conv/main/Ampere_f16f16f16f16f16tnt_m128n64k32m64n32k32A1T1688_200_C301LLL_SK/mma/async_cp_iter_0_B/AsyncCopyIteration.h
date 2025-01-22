#pragma once
#include <spconvlib/cumm/conv/main/Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK/mma/async_cp_iter_0_B/cp_async_copy/CpAsyncCopy.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK {
namespace mma {
namespace async_cp_iter_0_B {
using CpAsyncCp = spconvlib::cumm::conv::main::Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK::mma::async_cp_iter_0_B::cp_async_copy::CpAsyncCopy;
struct AsyncCopyIteration {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {
    
    ///// nothing to do here /////
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {
    
    ///// nothing to do here /////
  }
};
} // namespace async_cp_iter_0_B
} // namespace mma
} // namespace Ampere_f16f16f16f16f16tnt_m128n64k32m64n32k32A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib