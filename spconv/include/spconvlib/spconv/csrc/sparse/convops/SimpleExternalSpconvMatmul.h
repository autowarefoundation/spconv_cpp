#pragma once
#include <cublasLt.h>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/spconv/csrc/sparse/alloc/ExternalAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/ExternalSpconvMatmul.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using ExternalSpconvMatmul = spconvlib::spconv::csrc::sparse::convops::ExternalSpconvMatmul;
struct SimpleExternalSpconvMatmul : public ExternalSpconvMatmul {
  ExternalAllocator& alloc_;
  cublasLtHandle_t handle_ = 0;
  /**
   * @param alloc 
   */
   SimpleExternalSpconvMatmul(ExternalAllocator& alloc);
  
  virtual  ~SimpleExternalSpconvMatmul();
  /**
   * @param status 
   */
  static void check_cublas_status(cublasStatus_t status);
  /**
   * @param dtype 
   */
  static decltype(CUDA_R_16F) tv_dtype_to_blaslt(tv::DType dtype);
  inline static decltype(auto) tv_dtype_to_compute(tv::DType dtype)   {
    
    #if (CUDART_VERSION >= 11000)
      switch (dtype) {
      case tv::float32:
          return CUBLAS_COMPUTE_32F;
      case tv::float16:
          return CUBLAS_COMPUTE_16F;
      case tv::int32:
          return CUBLAS_COMPUTE_32I;
      case tv::int8:
          return CUBLAS_COMPUTE_32F;
      case tv::uint32:
          return CUBLAS_COMPUTE_32F;
      default:
          return CUBLAS_COMPUTE_32F;
      }
    #else
      switch (dtype) {
      case tv::float32:
          return CUDA_R_32F;
      case tv::float16:
          return CUDA_R_16F;
      case tv::int32:
          return CUDA_R_32I;
      case tv::int8:
          return CUDA_R_8I;
      case tv::uint32:
          return CUDA_R_32U;
      default:
          return CUDA_R_32F;
      }
    #endif
  }
  /**
   * @param handle 
   * @param stream 
   * @param a 
   * @param b 
   * @param c 
   * @param transA 
   * @param transB 
   */
  static void matmul_colmajor(cublasLtHandle_t handle, cudaStream_t stream, tv::Tensor a, tv::Tensor b, tv::Tensor c, bool transA, bool transB);
  /**
   * @param handle 
   * @param stream 
   * @param a 
   * @param b 
   * @param c 
   * @param transA 
   * @param transB 
   */
  static void matmul(cublasLtHandle_t handle, cudaStream_t stream, tv::Tensor a, tv::Tensor b, tv::Tensor c, bool transA, bool transB);
  /**
   * @param features_n 
   * @param filters_n 
   * @param all_weight_is_krsc 
   * @param is_kc_not_ck 
   * @param kv_center 
   * @param out_channel 
   * @param stream_int 
   */
  tv::Tensor indice_conv_init_gemm(std::string features_n, std::string filters_n, bool all_weight_is_krsc, bool is_kc_not_ck, int kv_center, int out_channel, std::uintptr_t stream_int);
};
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib