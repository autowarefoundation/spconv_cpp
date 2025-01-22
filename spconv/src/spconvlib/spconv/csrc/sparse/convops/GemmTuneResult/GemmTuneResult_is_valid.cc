#include <spconvlib/spconv/csrc/sparse/convops/GemmTuneResult.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using TensorView = spconvlib::cumm::common::TensorView;
bool GemmTuneResult::is_valid()   {
  
  return splitk > 0 && std::get<0>(arch) > 0;
}
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib