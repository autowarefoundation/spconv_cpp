#include <spconvlib/cumm/common/CompileInfo.h>
#include <spconvlib/cumm/common/_CudaInclude.h>
namespace spconvlib {
namespace cumm {
namespace common {
using _CudaInclude = spconvlib::cumm::common::_CudaInclude;
std::tuple<int, int> CompileInfo::get_compiled_cuda_version()   {
  
  #ifdef __CUDACC_VER_MAJOR__
  return std::make_tuple(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
  #else
  int ver = CUDA_VERSION; // from cuda.h
  return std::make_tuple(ver / 1000, (ver % 1000) / 10);
  #endif
}
} // namespace common
} // namespace cumm
} // namespace spconvlib