#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace gpSimt_f32f32f32f32f32ttt_m128n64k8m64n32k8A1_200_S1 {
namespace itera_p {
struct MaskTileIteratorParams {
  int32_t stride_;
  int64_t inc_strided_;
  int64_t inc_next_;
  int32_t const * indice_ptr_;
  __forceinline__ __host__ __device__  MaskTileIteratorParams()   {
    
  }
  __forceinline__ __host__ __device__  MaskTileIteratorParams(int stride, int32_t const * indice_ptr = nullptr) : stride_(stride), indice_ptr_(indice_ptr)  {
    
    inc_strided_ = stride * 16 * sizeof(float);
    inc_next_ = 32 - (7) *
                                    16 * stride *
                                    sizeof(float);
  }
};
} // namespace itera_p
} // namespace gpSimt_f32f32f32f32f32ttt_m128n64k8m64n32k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib