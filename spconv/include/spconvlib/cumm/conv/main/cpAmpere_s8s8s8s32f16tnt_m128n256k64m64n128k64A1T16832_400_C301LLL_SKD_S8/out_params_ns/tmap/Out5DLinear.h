#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_400_C301LLL_SKD_S8 {
namespace out_params_ns {
namespace tmap {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct Out5DLinear {
  __forceinline__ __host__ __device__ static tv::array<int, 2> initial_offset(int thread_idx)   {
    int warp_idx = thread_idx / 32;
    int lane_idx = thread_idx %  32;
    // Compute warp location
    int cluster_idx = warp_idx / 4;
    int residual_cluster = warp_idx % 4;
    int group_idx = residual_cluster / 2;
    int residual_group = residual_cluster % 2;
    int row_idx = residual_group / 1;
    int col_idx = residual_group % 1;
    // Compute per-lane offset
    // in 1d warp, row offset always 0
    int lane_row_offset = lane_idx / 8;
    int lane_col_offset = lane_idx % 8;
    // Compute coordinate in output space
    //
    // kPartShape: [Tile, Cluster, Group, Row, Col]
    // in out iter, x * 4 * 2 * 4 * 1 = x * 32
    // in smem loader, x * 4 * 1 * 1 * 1 = x * 4
    // 0, 4, 8, 12, full parallel,
    // in 0~3, warp 0 handle (0, 2), warp 1 handle (1, 3)
    // in 4~7, warp 2 handle (4, 6), warp 3 handle (5, 7)
    // so for thread 0, it handle [0, 0,32,64,96] and [2, 0,32,64,96]
    int cluster_offset = cluster_idx * 2 * 1 *
                        8 * 8;
    // 0, 1, 0, 1 * 8// warp 0 handle [0, 2], warp 1 handle [1, 3]
    int group_offset = group_idx * 8 * 8;
    // 0
    int row_offset = row_idx * 1 * 4; // 1d
    // we mul kElementsPerAccess here because unit of kAccessShape2D[1] isn't element.
    int column_offset =
        col_idx * 2 * 8 * 16;
    return {cluster_offset + group_offset + row_offset + lane_row_offset, 
        (column_offset + lane_col_offset) * 16};
  }
  __forceinline__ __host__ __device__ static tv::array<int64_t, 3> iteration_inc_params(int stride)   {
    tv::array<int64_t, 3> increments{};
    increments[0] = stride * 1 -
                    stride * 1 * (1 - 1) -
                    stride * 4 * (1 - 1);
    increments[1] =
        stride * 1 - stride * 4 * (1 - 1);
    increments[2] = stride * 4;
    return increments;
  }
  __forceinline__ __host__ __device__ static tv::array<int64_t, 4> iteration_advance_params(int stride)   {
    tv::array<int64_t, 4> advances{};
    // so advances[0] == 
    advances[0] =
        stride * 1 * 1 * 2 * 8;
    // TODO for cluster, advance_cluster should be wrong but the dilation of
    // cluster is always 1 in all cutlass configs. review this later.
    advances[1] = stride * 2 * 1 * 8 *
                8;
    // move to next 'dilation'
    // row dilation: LaneMmaShape::kM
    // for standard strided access,
    // first dilation, we need handle 0, 2, 4, 6 group, so warp0 = 0, 4; warp1 =
    // 2, 6, delta = 16 0, 2, 4, 6 rows in smem mapped to 0, 2, 4, 6 rows in
    // output memory (with different dilation). second dilation, we need handle
    // 1, 3, 5, 7 group, so warp0 = 1, 5; warp1 = 3, 7 1, 3, 5, 7 rows in smem
    // mapped to 1, 3, 5, 7 rows in output memory (with different dilation).
    // and advance_group should be part_shape[3] * part_dilation[3].
    // but the layout of lane mma result isn't standard.
    // they are strided too. columns are fixed in OutWarpTileIter, but rows
    // aren't. if lane mma count row is 2, we need to move to bottom part of mma
    // block instead of move forward.
    // first dilation, we need handle 0, 1, 2, 3 group, so warp0 = 0, 2; warp1 =
    // 1, 3 0, 2, 4, 6 rows in smem mapped to 0, 1, 2, 3 rows in output memory
    // second dilation, we need handle 4, 5, 6, 7 group, so warp0 = 4, 6; warp1
    // = 5, 7 1, 3, 5, 7 rows in smem mapped to 4, 5, 6, 7 rows in output memory
    advances[2] =
        stride * (2 - 1) * 8 * 8;
    advances[3] = stride * 8;
    return advances;
  }
};
} // namespace tmap
} // namespace out_params_ns
} // namespace cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_400_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib