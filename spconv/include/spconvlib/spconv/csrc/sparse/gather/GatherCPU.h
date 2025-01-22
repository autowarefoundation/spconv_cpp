#pragma once
#include <tensorview/parallel/all.h>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/cumm/common/GemmDTypes.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace gather {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmDTypes = spconvlib::cumm::common::GemmDTypes;
struct GatherCPU {
  /**
   * @param out 
   * @param in 
   * @param inds 
   */
  static void gather(tv::Tensor out, tv::Tensor in, tv::Tensor inds);
  /**
   * @param out 
   * @param in 
   * @param inds 
   */
  static void scatter_add(tv::Tensor out, tv::Tensor in, tv::Tensor inds);
};
} // namespace gather
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib