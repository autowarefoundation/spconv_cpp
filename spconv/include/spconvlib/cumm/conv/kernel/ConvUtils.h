#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace kernel {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct ConvUtils {
  TV_HOST_DEVICE_INLINE static tv::array<int, 3> get_spconv_logical_tile_count(int m, int n, int k, int tile_m, int tile_n, int split_k_slices, int kv, int op_type)   {
    
    tv::array<int, 3> grid_dims;
    if (op_type == 2){
        // n = C * kv
        int C = n / kv;
        // for wgrad, we need to ensure a block must be covered by one mask
        // so refined_n = tv::div_up(C, tile_n) * tile_n * kv
        // for example, C = 130, tile_n = 64, so one kernel loc need three
        // block 64 * 3 = 192, then refined_n = 192 * kv
        // n = tv::div_up(C, tile_n) * tile_n * kv;
        grid_dims[1] = tv::div_up(C, tile_n) * kv;
    }else{
        grid_dims[1] = tv::div_up(n, tile_n);
    }
    grid_dims[0] = tv::div_up(m, tile_m);
    grid_dims[2] = split_k_slices;
    return grid_dims;
  }
};
} // namespace kernel
} // namespace conv
} // namespace cumm
} // namespace spconvlib