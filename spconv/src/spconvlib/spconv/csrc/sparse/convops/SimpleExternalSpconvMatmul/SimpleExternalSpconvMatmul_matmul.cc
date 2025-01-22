#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using ExternalSpconvMatmul = spconvlib::spconv::csrc::sparse::convops::ExternalSpconvMatmul;
void SimpleExternalSpconvMatmul::matmul(cublasLtHandle_t handle, cudaStream_t stream, tv::Tensor a, tv::Tensor b, tv::Tensor c, bool transA, bool transB)   {
  
  return matmul_colmajor(handle, stream, b, a, c, transB, transA);
}
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib