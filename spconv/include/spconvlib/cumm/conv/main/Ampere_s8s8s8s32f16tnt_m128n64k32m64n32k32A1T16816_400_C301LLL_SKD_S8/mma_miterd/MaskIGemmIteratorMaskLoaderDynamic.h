#pragma once
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/common/GemmBasic.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f16tnt_m128n64k32m64n32k32A1T16816_400_C301LLL_SKD_S8 {
namespace mma_miterd {
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
struct MaskIGemmIteratorMaskLoaderDynamic {
  int current_mask;
  const int& mask_int_count;
  int mask_load_idx;
  int RS_pos;
  int RS_offset;
  const uint32_t & mask_filter;
  const int& tile_offset_m;
  const uint32_t* const& mask_ptr;
  uint32_t* const& mask_out_ptr;
  const int& RS;
  const bool& reverse;
  const int& gemm_k_iteration;
  const int& lane_idx;
  const int& problem_m;
  bool end;
  int mask_or_sum;
  __forceinline__ __device__  MaskIGemmIteratorMaskLoaderDynamic(const uint32_t* const& mask_ptr, uint32_t* const& mask_out_ptr, const int& mask_int_count, const int& tile_offset_m, const int& gemm_k_iteration, const int& RS, const uint32_t& mask_filter, const bool& reverse, const int& lane_idx, const int& problem_m) : mask_int_count(mask_int_count), mask_ptr(mask_ptr), mask_out_ptr(mask_out_ptr), tile_offset_m(tile_offset_m), RS(RS), gemm_k_iteration(gemm_k_iteration), end(false), reverse(reverse), mask_filter(mask_filter), lane_idx(lane_idx), problem_m(problem_m)  {
    
    mask_or_sum = 0;
    int used_rs = tv::div_up(RS, 32);
    if (reverse){
      for (mask_load_idx = 0; mask_load_idx < used_rs - 1; ++mask_load_idx){
          load_mask(true);
          mask_or_sum |= __brev(current_mask);
      }
    }
    else{
      for (mask_load_idx = 1; mask_load_idx < used_rs; ++mask_load_idx){
          load_mask(true);
          mask_or_sum |= current_mask;
      }
    }
    init_mask_iter(true);
    mask_or_sum |= current_mask;
    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    //     printf("OP ConvOpType.kForward into Dynamic Mask Loader  is increment k,   mask_int_count:%d, reverse: %d\n", mask_int_count, (int)reverse);
    // __syncwarp();
  }
  __device__ inline void load_mask(const bool& save = false)   {
    
    current_mask = 0;
    tv::array<uint32_t, 4> masks;
    masks.clear();
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i){
        if (tile_offset_m * 128 + i * 32 + lane_idx < problem_m){
            masks[i] = mask_ptr[mask_int_count * (tile_offset_m * 128 + i * 32 + lane_idx) + mask_load_idx];
        }
    }
    TV_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i){
        current_mask |= masks[i];
    }
    // perform a warp reduce to get block mask
    TV_PRAGMA_UNROLL
    for (int mask = 16; mask > 0; mask /= 2) {
        current_mask |= __shfl_xor_sync(0xffffffff, current_mask, mask, 32);
    }
    current_mask &= mask_filter;
    if (save){
      if (mask_out_ptr != nullptr)
          mask_out_ptr[tile_offset_m * mask_int_count + mask_load_idx] = current_mask;
    }
  }
  __device__ inline void init_mask_iter(const bool& save = false)   {
    
    mask_load_idx = 0;
    RS_pos = RS_offset = 0;
    load_mask(save);
  }
  __device__ inline void operator++()   {
    
    if (++RS_pos >= RS){
      end = true;
      return;
    }
    if ((RS_pos + RS_offset) % 32 == 0){
      ++mask_load_idx;
      load_mask();
    }
  }
  __device__ inline bool valid()   {
    
    return !!(
            current_mask & (1u << ((RS_pos + RS_offset) % 32))
            );
  }
  __device__ inline bool empty()   {
    
    return !mask_or_sum;
  }
};
} // namespace mma_miterd
} // namespace Ampere_s8s8s8s32f16tnt_m128n64k32m64n32k32A1T16816_400_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib