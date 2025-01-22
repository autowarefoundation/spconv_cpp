#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using ExternalSpconvMatmul = spconvlib::spconv::csrc::sparse::convops::ExternalSpconvMatmul;
decltype(CUDA_R_16F) SimpleExternalSpconvMatmul::tv_dtype_to_blaslt(tv::DType dtype)   {
  
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
}
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib