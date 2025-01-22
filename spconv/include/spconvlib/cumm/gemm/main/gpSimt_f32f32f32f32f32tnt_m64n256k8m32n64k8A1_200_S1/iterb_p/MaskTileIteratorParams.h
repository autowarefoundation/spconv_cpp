#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace gpSimt_f32f32f32f32f32tnt_m64n256k8m32n64k8A1_200_S1 {
namespace iterb_p {
struct MaskTileIteratorParams {
  int32_t stride_;
  int64_t inc_strided_;
  int64_t inc_next_;
  __forceinline__ __host__ __device__  MaskTileIteratorParams()   {
    
  }
  __forceinline__ __host__ __device__  MaskTileIteratorParams(int stride) : stride_(stride)  {
    
    inc_strided_ = stride * 32 * sizeof(float);
    inc_next_ = 32 - (7) *
                                    32 * stride *
                                    sizeof(float);
  }
};
} // namespace iterb_p
} // namespace gpSimt_f32f32f32f32f32tnt_m64n256k8m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib