#pragma once
#include <spconvlib/cumm/conv/main/cpTuring_f16f16f16f32f32tnt_m64n128k32m32n64k32A1T1688_200_C301LLL_SK/out_params_ns/tmap/Out5DLinear.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace cpTuring_f16f16f16f32f32tnt_m64n128k32m32n64k32A1T1688_200_C301LLL_SK {
namespace out_params_scalebias_ns {
using ThreadMap = spconvlib::cumm::conv::main::cpTuring_f16f16f16f32f32tnt_m64n128k32m32n64k32A1T1688_200_C301LLL_SK::out_params_ns::tmap::Out5DLinear;
struct OutIteratorParams {
  int64_t stride;
  int64_t increment_row;
  int64_t increment_group;
  int64_t increment_cluster;
  int64_t advance_row;
  int64_t advance_group;
  int64_t advance_cluster;
  int64_t advance_tile;
  __forceinline__ __host__ __device__  OutIteratorParams()   {
    
  }
  __forceinline__ __host__ __device__  OutIteratorParams(int stride_) : stride(stride_)  {
    
    auto increment_params = ThreadMap::iteration_inc_params(stride);
    auto advance_params = ThreadMap::iteration_advance_params(stride);
    increment_cluster = increment_params[0];
    increment_group = increment_params[1];
    increment_row = increment_params[2];
    advance_tile = advance_params[0];
    advance_cluster = advance_params[1];
    advance_group = advance_params[2];
    advance_row = advance_params[3];
  }
};
} // namespace out_params_scalebias_ns
} // namespace cpTuring_f16f16f16f32f32tnt_m64n128k32m32n64k32A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib