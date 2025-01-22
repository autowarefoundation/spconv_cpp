#include <spconvlib/cumm/common/CompileInfo.h>
namespace spconvlib {
namespace cumm {
namespace common {
bool CompileInfo::gemm_algo_can_use_ptx(std::tuple<int, int> min_arch, std::tuple<int, int> arch)   {
  
  return false;
}
} // namespace common
} // namespace cumm
} // namespace spconvlib