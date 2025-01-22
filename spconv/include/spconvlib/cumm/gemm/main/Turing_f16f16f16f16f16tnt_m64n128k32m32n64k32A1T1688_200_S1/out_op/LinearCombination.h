#pragma once
#include <tensorview/gemm/math/all.h>
#include <tensorview/gemm/core/constants.h>
#include <spconvlib/cumm/gemm/main/Turing_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T1688_200_S1/out_op/unaryop/UnaryActivation.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T1688_200_S1 {
namespace out_op {
using UnaryOp = spconvlib::cumm::gemm::main::Turing_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T1688_200_S1::out_op::unaryop::UnaryActivation;
struct LinearCombination {
  tv::half_t alpha;
  tv::half_t beta;
  tv::half_t act_alpha;
  tv::half_t act_beta;
  tv::gemm::Activation act_type;
  __forceinline__ __device__  LinearCombination(tv::half_t alpha = tv::half_t(1), tv::half_t beta = tv::half_t(0), tv::half_t act_alpha = tv::half_t(0), tv::half_t act_beta = tv::half_t(0), tv::gemm::Activation type = tv::gemm::Activation::kNone) : alpha(alpha), beta(beta), act_alpha(act_alpha), act_beta(act_beta), act_type(type)  {
    
  }
  __forceinline__ __device__ bool is_source_needed()  const {
    return  beta != tv::half_t(0);
  }
  __forceinline__ __device__ void set_k_partition(int k_part, int k_part_count)   {
    
    if (k_part) {
        beta = tv::half_t(1);
    }
  }
  __forceinline__ __device__ tv::array<tv::half_t, 8, 0> operator()(tv::array<tv::half_t, 8, 0> const& accumulator, tv::array<tv::half_t, 8, 0> const& source)  const {
    
    tv::gemm::NumericArrayConverter<tv::half_t, tv::half_t, 8, tv::gemm::FloatRoundStyle::round_to_nearest>
        source_converter;
    tv::gemm::NumericArrayConverter<tv::half_t, tv::half_t, 8, tv::gemm::FloatRoundStyle::round_to_nearest>
        accumulator_converter;
    tv::array<tv::half_t, 8, 0> converted_source = source_converter(source);
    tv::array<tv::half_t, 8, 0> converted_accumulator = accumulator_converter(accumulator);
    tv::array<tv::half_t, 8, 0> intermediate;
    tv::math::multiplies<tv::array<tv::half_t, 8, 0>> mul_add_source;
    tv::math::multiply_add<tv::array<tv::half_t, 8, 0>> mul_add_accumulator;
    intermediate =
        mul_add_source(beta, converted_source); // X =  beta * C + uniform
    intermediate = mul_add_accumulator(alpha, converted_accumulator,
                                    intermediate); // D = alpha * Accum + X
    UnaryOp op;
    intermediate = op(intermediate, act_type, act_alpha, act_beta);
    // Convert to destination numeric type
    tv::gemm::NumericArrayConverter<tv::half_t, tv::half_t, 8, tv::gemm::FloatRoundStyle::round_to_nearest>
        destination_converter;
    return destination_converter(intermediate);
  }
  __forceinline__ __device__ tv::array<tv::half_t, 8, 0> operator()(tv::array<tv::half_t, 8, 0> const& accumulator)  const {
    
    tv::gemm::NumericArrayConverter<tv::half_t, tv::half_t, 8, tv::gemm::FloatRoundStyle::round_to_nearest>
        accumulator_converter;
    tv::array<tv::half_t, 8, 0> converted_accumulator = accumulator_converter(accumulator);
    tv::array<tv::half_t, 8, 0> intermediate;
    tv::math::multiplies<tv::array<tv::half_t, 8, 0>> mul_accumulator;
    intermediate = mul_accumulator(alpha, converted_accumulator); // D = alpha * Accum + X
    UnaryOp op;
    intermediate = op(intermediate, act_type, act_alpha, act_beta);
    // Convert to destination numeric type
    tv::gemm::NumericArrayConverter<tv::half_t, tv::half_t, 8, tv::gemm::FloatRoundStyle::round_to_nearest>
        destination_converter;
    return destination_converter(intermediate);
  }
};
} // namespace out_op
} // namespace Turing_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T1688_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib