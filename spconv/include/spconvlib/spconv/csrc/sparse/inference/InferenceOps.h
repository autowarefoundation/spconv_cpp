#pragma once
#include <tensorview/gemm/core/constants.h>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/spconv/csrc/utils/launch/LaunchUtils.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace inference {
using TensorView = spconvlib::cumm::common::TensorView;
using LaunchUtils = spconvlib::spconv::csrc::utils::launch::LaunchUtils;
struct InferenceOps {
  static constexpr int kMaxGridYZDim = 65535;
  /**
   * @param out 
   * @param bias 
   * @param act_type 
   * @param alpha 
   * @param beta 
   * @param stream 
   */
  static void bias_add_act_inplace(tv::Tensor out, tv::Tensor bias, tv::gemm::Activation act_type = tv::gemm::Activation::kNone, float alpha = 0.0, float beta = 0.0, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param bias 
   * @param stream 
   */
  static void bias_add_inplace(tv::Tensor out, tv::Tensor bias, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param act_type 
   * @param alpha 
   * @param beta 
   * @param stream 
   */
  static void activation_inplace(tv::Tensor out, tv::gemm::Activation act_type, float alpha, float beta, std::uintptr_t stream = 0);
};
} // namespace inference
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib