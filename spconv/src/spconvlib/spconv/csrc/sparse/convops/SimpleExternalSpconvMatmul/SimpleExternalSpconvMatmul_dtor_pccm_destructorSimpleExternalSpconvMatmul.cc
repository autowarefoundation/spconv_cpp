#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using ExternalSpconvMatmul = spconvlib::spconv::csrc::sparse::convops::ExternalSpconvMatmul;
 SimpleExternalSpconvMatmul::~SimpleExternalSpconvMatmul()   {
  
  if (handle_){
      cublasLtDestroy(handle_);
  }
}
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib