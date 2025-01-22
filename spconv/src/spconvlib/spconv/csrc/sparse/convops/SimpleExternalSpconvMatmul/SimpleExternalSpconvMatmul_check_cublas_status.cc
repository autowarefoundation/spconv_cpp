#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using ExternalSpconvMatmul = spconvlib::spconv::csrc::sparse::convops::ExternalSpconvMatmul;
void SimpleExternalSpconvMatmul::check_cublas_status(cublasStatus_t status)   {
  
  if (status != CUBLAS_STATUS_SUCCESS) {
      printf("cuBLAS API failed with status %d\n", status);
      throw std::logic_error("cuBLAS API failed");
  }
}
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib