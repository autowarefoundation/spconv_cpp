#pragma once
#include <tensorview/parallel/all.h>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/cumm/common/GemmDTypes.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace maxpool {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmDTypes = spconvlib::cumm::common::GemmDTypes;
struct IndiceMaxPoolCPU {
  /**
   * @param out_indices 
   * @param coords 
   * @param counts 
   */
  static void global_pool_rearrange(tv::Tensor out_indices, tv::Tensor coords, tv::Tensor counts);
  /**
   * @param out 
   * @param in 
   * @param out_inds 
   * @param in_inds 
   * @param stream 
   */
  static void forward(tv::Tensor out, tv::Tensor in, tv::Tensor out_inds, tv::Tensor in_inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param in 
   * @param dout 
   * @param din 
   * @param out_inds 
   * @param in_inds 
   * @param stream 
   */
  static void backward(tv::Tensor out, tv::Tensor in, tv::Tensor dout, tv::Tensor din, tv::Tensor out_inds, tv::Tensor in_inds, std::uintptr_t stream = 0);
};
} // namespace maxpool
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib