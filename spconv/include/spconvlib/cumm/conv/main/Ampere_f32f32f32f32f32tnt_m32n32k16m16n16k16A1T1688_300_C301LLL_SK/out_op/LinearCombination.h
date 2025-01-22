#pragma once
#include <tensorview/gemm/math/all.h>
#include <tensorview/gemm/core/constants.h>
#include <spconvlib/cumm/conv/main/Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK/out_op/unaryop/UnaryActivation.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK {
namespace out_op {
using UnaryOp = spconvlib::cumm::conv::main::Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK::out_op::unaryop::UnaryActivation;
struct LinearCombination {
  float alpha;
  float beta;
  float act_alpha;
  float act_beta;
  tv::gemm::Activation act_type;
  __forceinline__ __device__  LinearCombination(float alpha = float(1), float beta = float(0), float act_alpha = float(0), float act_beta = float(0), tv::gemm::Activation type = tv::gemm::Activation::kNone) : alpha(alpha), beta(beta), act_alpha(act_alpha), act_beta(act_beta), act_type(type)  {
    
  }
  __forceinline__ __device__ bool is_source_needed()  const {
    return  beta != float(0);
  }
  __forceinline__ __device__ void set_k_partition(int k_part, int k_part_count)   {
    
    if (k_part) {
        beta = float(1);
    }
  }
  __forceinline__ __device__ tv::array<float, 4, 0> operator()(tv::array<float, 4, 0> const& accumulator, tv::array<float, 4, 0> const& source)  const {
    
    tv::gemm::NumericArrayConverter<float, float, 4, tv::gemm::FloatRoundStyle::round_to_nearest>
        source_converter;
    tv::gemm::NumericArrayConverter<float, float, 4, tv::gemm::FloatRoundStyle::round_to_nearest>
        accumulator_converter;
    tv::array<float, 4, 0> converted_source = source_converter(source);
    tv::array<float, 4, 0> converted_accumulator = accumulator_converter(accumulator);
    tv::array<float, 4, 0> intermediate;
    tv::math::multiplies<tv::array<float, 4, 0>> mul_add_source;
    tv::math::multiply_add<tv::array<float, 4, 0>> mul_add_accumulator;
    intermediate =
        mul_add_source(beta, converted_source); // X =  beta * C + uniform
    intermediate = mul_add_accumulator(alpha, converted_accumulator,
                                    intermediate); // D = alpha * Accum + X
    UnaryOp op;
    intermediate = op(intermediate, act_type, act_alpha, act_beta);
    // Convert to destination numeric type
    tv::gemm::NumericArrayConverter<float, float, 4, tv::gemm::FloatRoundStyle::round_to_nearest>
        destination_converter;
    return destination_converter(intermediate);
  }
  __forceinline__ __device__ tv::array<float, 4, 0> operator()(tv::array<float, 4, 0> const& accumulator)  const {
    
    tv::gemm::NumericArrayConverter<float, float, 4, tv::gemm::FloatRoundStyle::round_to_nearest>
        accumulator_converter;
    tv::array<float, 4, 0> converted_accumulator = accumulator_converter(accumulator);
    tv::array<float, 4, 0> intermediate;
    tv::math::multiplies<tv::array<float, 4, 0>> mul_accumulator;
    intermediate = mul_accumulator(alpha, converted_accumulator); // D = alpha * Accum + X
    UnaryOp op;
    intermediate = op(intermediate, act_type, act_alpha, act_beta);
    // Convert to destination numeric type
    tv::gemm::NumericArrayConverter<float, float, 4, tv::gemm::FloatRoundStyle::round_to_nearest>
        destination_converter;
    return destination_converter(intermediate);
  }
};
} // namespace out_op
} // namespace Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib