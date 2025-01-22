#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/conv/main/Simt_f32f32f32f32f32tnt_m64n128k8m32n64k8A1_200_C301LLL_SK/out_op/LinearCombination.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m64n128k8m32n64k8A1_200_C301LLL_SK {
namespace output {
namespace out_ns_apply {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using OutputOp = spconvlib::cumm::conv::main::Simt_f32f32f32f32f32tnt_m64n128k8m32n64k8A1_200_C301LLL_SK::out_op::LinearCombination;
struct ApplyOutputOp {
  __forceinline__ __device__ static void apply_output_operator(tv::array<float, 8, 0> & output_fragment, OutputOp const & output_op, tv::array<float, 8, 0> const & aligned_accum_fragment, tv::array<float, 8, 0> const & source_fragment)   {
    
    constexpr int kOutFragCount = tv::array_size_v<tv::array<float, 8, 0>>;
    using OutAccessType = tv::array<typename tv::array<float, 8, 0>::value_type, 1, 0>;
    using InputAccessType = tv::array<typename tv::array<float, 8, 0>::value_type, 1, 0>;
    OutAccessType *output_frag_ptr =
        reinterpret_cast<OutAccessType *>(&output_fragment);
    InputAccessType const *compute_frag_ptr =
        reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
    OutAccessType const *source_frag_ptr =
        reinterpret_cast<OutAccessType const *>(&source_fragment);
    constexpr int kOutOpIterations = kOutFragCount / 1;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < kOutOpIterations; ++i) {
        output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
    }
  }
  __forceinline__ __device__ static void apply_output_operator_no_source(tv::array<float, 8, 0> & output_fragment, OutputOp const & output_op, tv::array<float, 8, 0> const & aligned_accum_fragment)   {
    
    constexpr int kOutFragCount = tv::array_size_v<tv::array<float, 8, 0>>;
    using OutAccessType = tv::array<typename tv::array<float, 8, 0>::value_type, 1, 0>;
    using InputAccessType = tv::array<typename tv::array<float, 8, 0>::value_type, 1, 0>;
    OutAccessType *output_frag_ptr =
        reinterpret_cast<OutAccessType *>(&output_fragment);
    InputAccessType const *compute_frag_ptr =
        reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
    constexpr int kOutOpIterations = kOutFragCount / 1;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < kOutOpIterations; ++i) {
        output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
    }
  }
};
} // namespace out_ns_apply
} // namespace output
} // namespace Simt_f32f32f32f32f32tnt_m64n128k8m32n64k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib