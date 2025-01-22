#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace inference {
using TensorView = spconvlib::cumm::common::TensorView;
using LaunchUtils = spconvlib::spconv::csrc::utils::launch::LaunchUtils;
void InferenceOps::bias_add_inplace(tv::Tensor out, tv::Tensor bias, std::uintptr_t stream)   {
  
  return bias_add_act_inplace(out, bias, tv::gemm::Activation::kNone, 0, 0, stream);
}
} // namespace inference
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib