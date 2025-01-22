#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/conv/main/Turing_f16f16f16f16f16tnt_m32n128k64m32n64k64A1T1688_200_C301LLL_SK/out_op/LinearCombination.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_f16f16f16f16f16tnt_m32n128k64m32n64k64A1T1688_200_C301LLL_SK {
namespace output {
namespace out_ns_apply {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using OutputOp = spconvlib::cumm::conv::main::Turing_f16f16f16f16f16tnt_m32n128k64m32n64k64A1T1688_200_C301LLL_SK::out_op::LinearCombination;
struct ApplyOutputOp {
  __forceinline__ __device__ static void apply_output_operator(tv::array<tv::half_t, 16, 0> & output_fragment, OutputOp const & output_op, tv::array<tv::half_t, 16, 0> const & aligned_accum_fragment, tv::array<tv::half_t, 16, 0> const & source_fragment)   {
    
    constexpr int kOutFragCount = tv::array_size_v<tv::array<tv::half_t, 16, 0>>;
    using OutAccessType = tv::array<typename tv::array<tv::half_t, 16, 0>::value_type, 8, 0>;
    using InputAccessType = tv::array<typename tv::array<tv::half_t, 16, 0>::value_type, 8, 0>;
    OutAccessType *output_frag_ptr =
        reinterpret_cast<OutAccessType *>(&output_fragment);
    InputAccessType const *compute_frag_ptr =
        reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
    OutAccessType const *source_frag_ptr =
        reinterpret_cast<OutAccessType const *>(&source_fragment);
    constexpr int kOutOpIterations = kOutFragCount / 8;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < kOutOpIterations; ++i) {
        output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
    }
  }
  __forceinline__ __device__ static void apply_output_operator_no_source(tv::array<tv::half_t, 16, 0> & output_fragment, OutputOp const & output_op, tv::array<tv::half_t, 16, 0> const & aligned_accum_fragment)   {
    
    constexpr int kOutFragCount = tv::array_size_v<tv::array<tv::half_t, 16, 0>>;
    using OutAccessType = tv::array<typename tv::array<tv::half_t, 16, 0>::value_type, 8, 0>;
    using InputAccessType = tv::array<typename tv::array<tv::half_t, 16, 0>::value_type, 8, 0>;
    OutAccessType *output_frag_ptr =
        reinterpret_cast<OutAccessType *>(&output_fragment);
    InputAccessType const *compute_frag_ptr =
        reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
    constexpr int kOutOpIterations = kOutFragCount / 8;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < kOutOpIterations; ++i) {
        output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
    }
  }
};
} // namespace out_ns_apply
} // namespace output
} // namespace Turing_f16f16f16f16f16tnt_m32n128k64m32n64k64A1T1688_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib