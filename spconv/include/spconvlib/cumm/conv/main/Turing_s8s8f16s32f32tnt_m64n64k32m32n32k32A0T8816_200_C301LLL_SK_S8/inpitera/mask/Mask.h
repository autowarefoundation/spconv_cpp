#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8f16s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8 {
namespace inpitera {
namespace mask {
struct Mask {
  tv::array<uint32_t, 1> mask_;
  __forceinline__ __host__ __device__  Mask()   {
    
  }
  __forceinline__ __host__ __device__ void clear()   {
    
    mask_.clear();
  }
  __forceinline__ __host__ __device__ uint32_t query(int idx)  const {
    
    return mask_[0] & (1 << idx);
  }
  __forceinline__ __host__ __device__ uint32_t query_coord(int idx0, int idx1, int idx2, int idx3)  const {
    
    return query(idx0 * 16 + idx1 * 16 + idx2 * 16 + idx3 * 1);
  }
  __forceinline__ __host__ __device__ void set(uint32_t pred, int idx)   {
    
    mask_[0] |= pred << idx;
  }
  __forceinline__ __host__ __device__ void set_coord(uint32_t pred, int idx0, int idx1, int idx2, int idx3)   {
    
    return set(pred, idx0 * 16 + idx1 * 16 + idx2 * 16 + idx3 * 1);
  }
  __forceinline__ __host__ __device__ void clear_if_pred(uint32_t pred, int idx0, int idx1, int idx2, int idx3)   {
    
    return set(pred, idx0 * 16 + idx1 * 16 + idx2 * 16);
  }
  __forceinline__ __device__ void clear_all_mask_if_not_pred(bool pred)   {
    
    mask_[0] = pred ? mask_[0] : 0u;
  }
  __forceinline__ __device__ void clear_all_mask_if_pred(bool pred)   {
    
    mask_[0] = pred ? 0u : mask_[0];
  }
};
} // namespace mask
} // namespace inpitera
} // namespace Turing_s8s8f16s32f32tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib