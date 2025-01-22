#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using ExternalSpconvMatmul = spconvlib::spconv::csrc::sparse::convops::ExternalSpconvMatmul;
void SimpleExternalSpconvMatmul::matmul_colmajor(cublasLtHandle_t handle, cudaStream_t stream, tv::Tensor a, tv::Tensor b, tv::Tensor c, bool transA, bool transB)   {
  
    bool transC = false;
    auto m = a.dim(int(!transA));
    auto k = a.dim(int(transA));
    auto k2 = b.dim(int(!transB));
    auto n = b.dim(int(transB));
    TV_ASSERT_INVALID_ARG(k == k2, "error");
    TV_ASSERT_INVALID_ARG(a.dtype() == b.dtype(), "error");
    tv::TensorShape c_shape;
    if (transC) {
      c_shape = {m, n};
    } else {
      c_shape = {n, m};
    }
    if (c.empty()) {
      c = tv::Tensor(c_shape, a.dtype(), a.device());
    } else {
      TV_ASSERT_INVALID_ARG(c.dim(0) == c_shape[0] && c.dim(1) == c_shape[1],
                            "error");
    }
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    decltype(CUDA_R_16F) scalarType = CUDA_R_16F;
  #if CUDART_VERSION >= 11000
    decltype(CUBLAS_COMPUTE_32F) computeType = CUBLAS_COMPUTE_32F;
  #endif
    if (a.dtype() == tv::float16 && b.dtype() == tv::float16 &&
        c.dtype() == tv::float16) {
      scalarType = CUDA_R_16F;
  #if CUDART_VERSION >= 11000
      computeType = CUBLAS_COMPUTE_16F;
  #endif
    } else if (a.dtype() == tv::float32 && b.dtype() == tv::float32 &&
              c.dtype() == tv::float16) {
      scalarType = CUDA_R_32F;
  #if CUDART_VERSION >= 11000
      computeType = CUBLAS_COMPUTE_32F;
  #endif
    } else if (a.dtype() == tv::float32 && b.dtype() == tv::float32 &&
              c.dtype() == tv::float32) {
      scalarType = CUDA_R_32F;
  #if CUDART_VERSION >= 11000
      computeType = CUBLAS_COMPUTE_32F;
  #endif
    } else if (a.dtype() == tv::float16 && b.dtype() == tv::float16 &&
              c.dtype() == tv::float32) {
      scalarType = CUDA_R_32F;
  #if CUDART_VERSION >= 11000
      computeType = CUBLAS_COMPUTE_32F;
  #endif
    } else {
      TV_THROW_RT_ERR("unsupported");
    }
  #if CUDART_VERSION >= 11000
    check_cublas_status(
        cublasLtMatmulDescCreate(&operationDesc, computeType, scalarType));
  #else
    check_cublas_status(cublasLtMatmulDescCreate(&operationDesc, scalarType));
  #endif
    cublasOperation_t transa = !transA ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transb = !transB ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transc = !transC ? CUBLAS_OP_N : CUBLAS_OP_T;
    check_cublas_status(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    check_cublas_status(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    //   check_cublas_status(cublasLtMatmulDescSetAttribute(
    //       operationDesc, CUBLASLT_MATMUL_DESC_TRANSC, &transc,
    //       sizeof(transc)));
    check_cublas_status(cublasLtMatrixLayoutCreate(
        &Adesc, tv_dtype_to_blaslt(a.dtype()), transa == CUBLAS_OP_N ? m : k,
        transa == CUBLAS_OP_N ? k : m, a.stride(0)));
    check_cublas_status(cublasLtMatrixLayoutCreate(
        &Bdesc, tv_dtype_to_blaslt(b.dtype()), transb == CUBLAS_OP_N ? k : n,
        transb == CUBLAS_OP_N ? n : k, b.stride(0)));
    //   check_cublas_status(cublasLtMatrixLayoutCreate(
    //       &Cdesc, tv_dtype_to_blaslt(c.dtype()), transc == CUBLAS_OP_N ? m : n,
    //       transc == CUBLAS_OP_N ? n : m, c.dim(0)));
    check_cublas_status(cublasLtMatrixLayoutCreate(
        &Cdesc, tv_dtype_to_blaslt(c.dtype()), m, n, c.stride(0)));
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasLtMatmulPreference_t preference = NULL;
    check_cublas_status(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspaceSize = 0;
    check_cublas_status(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
    int returnedResults = 0;
    check_cublas_status(cublasLtMatmulAlgoGetHeuristic(
        handle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
        &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
      check_cublas_status(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    int alpha_storage[4];
    int beta_storage[4];
    if (scalarType == CUDA_R_32F) {
      *(reinterpret_cast<float *>(alpha_storage)) = 1.0f;
      *(reinterpret_cast<float *>(beta_storage)) = 0.0f;
    } else if (scalarType == CUDA_R_16F) {
      *(reinterpret_cast<__half *>(alpha_storage)) = __half(1.0f);
      *(reinterpret_cast<__half *>(beta_storage)) = __half(0.0f);
    } else {
      TV_THROW_RT_ERR("unsupported");
    }
    check_cublas_status(cublasLtMatmul(
        handle, operationDesc, alpha_storage, a.const_raw_data(), Adesc, b.const_raw_data(),
        Bdesc, beta_storage, c.raw_data(), Cdesc, c.raw_data(), Cdesc,
        &heuristicResult.algo, nullptr, 0, stream));
    if (preference)
      check_cublas_status(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc)
      check_cublas_status(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      check_cublas_status(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      check_cublas_status(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc)
      check_cublas_status(cublasLtMatmulDescDestroy(operationDesc));
    return;
}
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib