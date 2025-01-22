#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/gemm/layout/RowMajor.h>
#include <spconvlib/cumm/gemm/layout/ColumnMajor.h>
#include <spconvlib/cumm/gemm/main/Simt_f32f32f32f32f32tnt_m64n32k16m32n32k8A1_200_S1/mma/mma_ns_wmma/instmma/InstMma.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m64n32k16m32n32k8A1_200_S1 {
namespace mma {
namespace mma_ns_wmma {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using RowMajor = spconvlib::cumm::gemm::layout::RowMajor;
using ColumnMajor = spconvlib::cumm::gemm::layout::ColumnMajor;
using InstMma = spconvlib::cumm::gemm::main::Simt_f32f32f32f32f32tnt_m64n32k16m32n32k8A1_200_S1::mma::mma_ns_wmma::instmma::InstMma;
struct WarpMmaSimt {
  __forceinline__ __device__ void operator()(tv::array<float, 32, 0>& D, tv::array<float, 8, 0> const & A, tv::array<float, 4, 0> const & B, tv::array<float, 32, 0> const & C)   {
    
    constexpr ColumnMajor layoutA = ColumnMajor::from_shape({8, 1});
    constexpr RowMajor layoutB = RowMajor::from_shape({1, 4});
    constexpr RowMajor layoutC = RowMajor::from_shape({8, 4});
    InstMma mma;
    D = C;
    TV_PRAGMA_UNROLL
    for (int k = 0; k < 1; ++k) {
        TV_PRAGMA_UNROLL
        for (int n = 0; n < 4; ++n) {
            TV_PRAGMA_UNROLL
            for (int m = 0; m < 8; ++m) {
                // what's this????
                // Column-major serpentine sequence to maximize reuse of A operand.
                // "mma_tensor_op_sm70.h:243"
                int m_serpentine = (n % 2) ? (8 - 1 - m) : m;
                tv::array<float, 1, 0> d;
                tv::array<float, 1, 0> a;
                tv::array<float, 1, 0> b;
                d[0] = D[layoutC(m_serpentine, n)];
                a[0] = A[layoutA(m_serpentine, k)];
                b[0] = B[layoutB(k, n)];
                mma(d, a, b, d);
                D[layoutC(m_serpentine, n)] = d[0];
            }
        }
    }
  }
};
} // namespace mma_ns_wmma
} // namespace mma
} // namespace Simt_f32f32f32f32f32tnt_m64n32k16m32n32k8A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib