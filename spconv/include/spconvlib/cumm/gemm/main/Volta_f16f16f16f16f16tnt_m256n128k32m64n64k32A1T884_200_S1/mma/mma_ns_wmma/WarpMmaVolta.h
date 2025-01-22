#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/gemm/main/Volta_f16f16f16f16f16tnt_m256n128k32m64n64k32A1T884_200_S1/mma/mma_ns_wmma/tensorop/MmaSync.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Volta_f16f16f16f16f16tnt_m256n128k32m64n64k32A1T884_200_S1 {
namespace mma {
namespace mma_ns_wmma {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using InstMma = spconvlib::cumm::gemm::main::Volta_f16f16f16f16f16tnt_m256n128k32m64n64k32A1T884_200_S1::mma::mma_ns_wmma::tensorop::MmaSync;
struct WarpMmaVolta {
  __forceinline__ __device__ void operator()(tv::array<tv::half_t, 128, 0>& D, tv::array<tv::half_t, 16, 0> const & A, tv::array<tv::half_t, 16, 0> const & B, tv::array<tv::half_t, 128, 0> const & C)   {
    
    InstMma mma;
    D = C;
    tv::array<tv::half_t, 4, 0> const *ptr_A = reinterpret_cast<tv::array<tv::half_t, 4, 0> const *>(&A);
    tv::array<tv::half_t, 4, 0> const *ptr_B = reinterpret_cast<tv::array<tv::half_t, 4, 0> const *>(&B);
    tv::array<tv::half_t, 8, 0> *ptr_D = reinterpret_cast<tv::array<tv::half_t, 8, 0> *>(&D);
    TV_PRAGMA_UNROLL
    for (int outer_col = 0; outer_col < 2; ++outer_col) {
        TV_PRAGMA_UNROLL
        for (int inner_col = 0; inner_col < 2; ++inner_col) {
            TV_PRAGMA_UNROLL
            for (int outer_row = 0; outer_row < 2;
                ++outer_row) {
                TV_PRAGMA_UNROLL
                for (int inner_row = 0; inner_row < 2; ++inner_row) {
                    int op_col = inner_col + 2 * outer_col;
                    // Column-major serpentine sequence to maximize reuse of A operand.
                    int inner_row_serp = inner_row;
                    int outer_row_serp = outer_row;
                    if (op_col & 1) {
                        inner_row_serp = 2 - inner_row - 1;
                        outer_row_serp = 2 - outer_row - 1;
                    }
                    int op_row = inner_row_serp + 2 * outer_row_serp;
                    // op_idx: [kMmaTileIterations[1], kMmaTileIterations[0],
                    // kMmaIterations[1], kMmaIterations[0]]
                    int op_idx =
                        inner_row_serp +
                        2 *
                            (inner_col +
                            2 *
                                (outer_row_serp + 2 * outer_col));
                    mma(ptr_D[op_idx], ptr_A[op_row], ptr_B[op_col], ptr_D[op_idx]);
                }
            }
        }
    }
  }
};
} // namespace mma_ns_wmma
} // namespace mma
} // namespace Volta_f16f16f16f16f16tnt_m256n128k32m64n64k32A1T884_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib