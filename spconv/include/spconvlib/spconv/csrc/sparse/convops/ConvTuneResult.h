#pragma once
#include <spconvlib/cumm/common/GemmBasicHost.h>
#include <spconvlib/cumm/common/TensorView.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using TensorView = spconvlib::cumm::common::TensorView;
struct ConvTuneResult {
  tv::gemm::ConvAlgoDesp algo_desp;
  std::tuple<int, int> arch;
  int splitk;
  
   ConvTuneResult();
  /**
   * @param algo_desp 
   * @param arch 
   * @param splitk 
   */
   ConvTuneResult(tv::gemm::ConvAlgoDesp algo_desp, std::tuple<int, int> arch, int splitk);
  
  bool is_valid();
};
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib