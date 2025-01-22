#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace gpSimt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1 {
namespace iterb_p {
struct MaskTileIteratorParams {
  int32_t stride_;
  int64_t inc_strided_;
  int64_t inc_next_;
  __forceinline__ __host__ __device__  MaskTileIteratorParams()   {
    
  }
  __forceinline__ __host__ __device__  MaskTileIteratorParams(int stride) : stride_(stride)  {
    
    inc_strided_ = stride * 16 * sizeof(tv::half_t);
    inc_next_ = 32 - (7) *
                                    16 * stride *
                                    sizeof(tv::half_t);
  }
};
} // namespace iterb_p
} // namespace gpSimt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib