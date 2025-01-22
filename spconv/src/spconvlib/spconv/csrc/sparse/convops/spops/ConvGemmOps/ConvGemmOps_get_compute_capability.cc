#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
namespace spops {
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using GemmTuneResult = spconvlib::spconv::csrc::sparse::convops::GemmTuneResult;
using ConvTuneResult = spconvlib::spconv::csrc::sparse::convops::ConvTuneResult;
using ExternalSpconvMatmul = spconvlib::spconv::csrc::sparse::convops::ExternalSpconvMatmul;
using InferenceOps = spconvlib::spconv::csrc::sparse::inference::InferenceOps;
using GemmTuner = spconvlib::spconv::csrc::sparse::convops::gemmops::GemmTunerSimple;
using ConvTuner = spconvlib::spconv::csrc::sparse::convops::convops::ConvTunerSimple;
std::tuple<int, int> ConvGemmOps::get_compute_capability(int index)   {
  
  if (index == -1){
      checkCudaErrors(cudaGetDevice(&index));
  }
  #ifdef TV_CUDA
      cudaDeviceProp prop;
      checkCudaErrors(cudaGetDeviceProperties(&prop, index));
      return std::make_tuple(prop.major, prop.minor);
  #else 
      return std::make_tuple(-1, -1);
  #endif
}
} // namespace spops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib