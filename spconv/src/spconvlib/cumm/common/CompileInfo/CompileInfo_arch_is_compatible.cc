#include <spconvlib/cumm/common/CompileInfo.h>
namespace spconvlib {
namespace cumm {
namespace common {
bool CompileInfo::arch_is_compatible(std::tuple<int, int> arch)   {
  
  return arch_is_compiled(arch);
}
} // namespace common
} // namespace cumm
} // namespace spconvlib