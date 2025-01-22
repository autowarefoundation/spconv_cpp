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
struct GemmTuneResult {
  tv::gemm::GemmAlgoDesp algo_desp;
  std::tuple<int, int> arch;
  int splitk;
  
  bool is_valid();
  
   GemmTuneResult();
  /**
   * @param algo_desp 
   * @param arch 
   * @param splitk 
   */
   GemmTuneResult(tv::gemm::GemmAlgoDesp algo_desp, std::tuple<int, int> arch, int splitk);
};
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib