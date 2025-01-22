#include <spconvlib/cumm/common/CompileInfo.h>
namespace spconvlib {
namespace cumm {
namespace common {
std::vector<std::tuple<int, int>> CompileInfo::get_compiled_cuda_arch()   {
  
  std::vector<std::tuple<int, int>> res;
  res.push_back(std::make_tuple(7, 5));
  res.push_back(std::make_tuple(8, 0));
  res.push_back(std::make_tuple(8, 6));
  res.push_back(std::make_tuple(8, 9));
  return res;
}
} // namespace common
} // namespace cumm
} // namespace spconvlib