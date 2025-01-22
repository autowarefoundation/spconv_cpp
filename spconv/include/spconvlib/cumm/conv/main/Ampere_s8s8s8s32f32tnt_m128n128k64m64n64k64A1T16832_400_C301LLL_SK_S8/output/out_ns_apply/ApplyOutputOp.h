#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f32tnt_m128n128k64m64n64k64A1T16832_400_C301LLL_SK_S8/out_op/Int8Inference.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f32tnt_m128n128k64m64n64k64A1T16832_400_C301LLL_SK_S8 {
namespace output {
namespace out_ns_apply {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using OutputOp = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f32tnt_m128n128k64m64n64k64A1T16832_400_C301LLL_SK_S8::out_op::Int8Inference;
struct ApplyOutputOp {
  __forceinline__ __device__ static void apply_output_with_int8_operator(tv::array<int8_t, 16, 0> & output_fragment, OutputOp const & output_op, tv::array<int32_t, 16, 0> const & aligned_accum_fragment, tv::array<int8_t, 16, 0> const & source_fragment, tv::array<float, 16, 0> const & int8_bias_fragment, tv::array<float, 16, 0> const & int8_scale_fragment)   {
    
    constexpr int kOutFragCount = tv::array_size_v<tv::array<int8_t, 16, 0>>;
    using OutAccessType = tv::array<typename tv::array<int8_t, 16, 0>::value_type, 8, 0>;
    using InputAccessType = tv::array<typename tv::array<int32_t, 16, 0>::value_type, 8, 0>;
    using ScaleAccessType = tv::array<typename tv::array<float, 16, 0>::value_type, 8, 0>;
    OutAccessType *output_frag_ptr =
        reinterpret_cast<OutAccessType *>(&output_fragment);
    InputAccessType const *compute_frag_ptr =
        reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
    OutAccessType const *source_frag_ptr =
        reinterpret_cast<OutAccessType const *>(&source_fragment);
    ScaleAccessType const *bias_ptr =
        reinterpret_cast<ScaleAccessType const *>(&int8_bias_fragment);
    ScaleAccessType const *scale_ptr =
        reinterpret_cast<ScaleAccessType const *>(&int8_scale_fragment);
    constexpr int kOutOpIterations = kOutFragCount / 8;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < kOutOpIterations; ++i) {
        output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i], bias_ptr[i], scale_ptr[i]);
    }
  }
  __forceinline__ __device__ static void apply_output_with_int8_operator_no_source(tv::array<int8_t, 16, 0> & output_fragment, OutputOp const & output_op, tv::array<int32_t, 16, 0> const & aligned_accum_fragment, tv::array<float, 16, 0> const & int8_bias_fragment, tv::array<float, 16, 0> const & int8_scale_fragment)   {
    
    constexpr int kOutFragCount = tv::array_size_v<tv::array<int8_t, 16, 0>>;
    using OutAccessType = tv::array<typename tv::array<int8_t, 16, 0>::value_type, 8, 0>;
    using InputAccessType = tv::array<typename tv::array<int32_t, 16, 0>::value_type, 8, 0>;
    using ScaleAccessType = tv::array<typename tv::array<float, 16, 0>::value_type, 8, 0>;
    OutAccessType *output_frag_ptr =
        reinterpret_cast<OutAccessType *>(&output_fragment);
    InputAccessType const *compute_frag_ptr =
        reinterpret_cast<InputAccessType const *>(&aligned_accum_fragment);
    ScaleAccessType const *bias_ptr =
        reinterpret_cast<ScaleAccessType const *>(&int8_bias_fragment);
    ScaleAccessType const *scale_ptr =
        reinterpret_cast<ScaleAccessType const *>(&int8_scale_fragment);
    constexpr int kOutOpIterations = kOutFragCount / 8;
    TV_PRAGMA_UNROLL
    for (int i = 0; i < kOutOpIterations; ++i) {
        output_frag_ptr[i] = output_op(compute_frag_ptr[i], bias_ptr[i], scale_ptr[i]);
    }
  }
};
} // namespace out_ns_apply
} // namespace output
} // namespace Ampere_s8s8s8s32f32tnt_m128n128k64m64n64k64A1T16832_400_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib