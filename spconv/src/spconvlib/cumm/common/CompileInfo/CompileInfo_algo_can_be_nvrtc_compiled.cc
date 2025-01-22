#include <spconvlib/cumm/common/CompileInfo.h>
namespace spconvlib {
namespace cumm {
namespace common {
bool CompileInfo::algo_can_be_nvrtc_compiled(std::tuple<int, int> min_arch)   {
  
  auto cuda_ver = get_compiled_cuda_version();
  if (cuda_ver >= std::make_tuple(11, 8)){
      return min_arch <= std::make_tuple(9, 0);
  }
  if (cuda_ver >= std::make_tuple(11, 1)){
      return min_arch <= std::make_tuple(8, 6);
  }
  if (cuda_ver >= std::make_tuple(11, 0)){
      return min_arch <= std::make_tuple(8, 0);
  }
  if (cuda_ver >= std::make_tuple(10, 2)){
      return min_arch <= std::make_tuple(7, 5);
  }
  return false;
}
} // namespace common
} // namespace cumm
} // namespace spconvlib