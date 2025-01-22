#include <spconvlib/spconv/csrc/sparse/convops/GemmTuneResult.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using TensorView = spconvlib::cumm::common::TensorView;
 GemmTuneResult::GemmTuneResult() : algo_desp(tv::gemm::GemmAlgoDesp()), arch(std::make_tuple(-1, -1)), splitk(-1)  {
  
}
 GemmTuneResult::GemmTuneResult(tv::gemm::GemmAlgoDesp algo_desp, std::tuple<int, int> arch, int splitk) : algo_desp(algo_desp), arch(arch), splitk(splitk)  {
  
}
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib