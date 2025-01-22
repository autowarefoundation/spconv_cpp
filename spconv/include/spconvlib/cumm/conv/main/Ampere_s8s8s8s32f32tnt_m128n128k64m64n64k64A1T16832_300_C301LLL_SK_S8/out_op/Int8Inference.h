#pragma once
#include <tensorview/gemm/math/all.h>
#include <tensorview/gemm/core/constants.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f32tnt_m128n128k64m64n64k64A1T16832_300_C301LLL_SK_S8/out_op/unaryop/UnaryActivation.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f32tnt_m128n128k64m64n64k64A1T16832_300_C301LLL_SK_S8 {
namespace out_op {
using UnaryOp = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f32tnt_m128n128k64m64n64k64A1T16832_300_C301LLL_SK_S8::out_op::unaryop::UnaryActivation;
struct Int8Inference {
  float alpha;
  float beta;
  float act_alpha;
  float act_beta;
  tv::gemm::Activation act_type;
  __forceinline__ __device__  Int8Inference(float alpha = float(1), float beta = float(0), float act_alpha = float(0), float act_beta = float(0), tv::gemm::Activation type = tv::gemm::Activation::kNone) : alpha(alpha), beta(beta), act_alpha(act_alpha), act_beta(act_beta), act_type(type)  {
    
  }
  __forceinline__ __device__ bool is_source_needed()  const {
    return beta != float(0);
  }
  __forceinline__ __device__ void set_k_partition(int k_part, int k_part_count)   {
    
    if (k_part) {
        // beta is for self reduce (part k from other wraps)
        beta = float(1);
    }
  }
  __forceinline__ __device__ tv::array<int8_t, 8, 0> operator()(tv::array<int32_t, 8, 0> const& accumulator, tv::array<int8_t, 8, 0> const& source, tv::array<float, 8, 0> const& bias, tv::array<float, 8, 0> const& scale)  const {
    
    tv::gemm::NumericArrayConverter<float, int8_t, 8, tv::gemm::FloatRoundStyle::round_to_nearest>
        source_converter;
    tv::array<float, 8, 0> converted_source = source_converter(source);
    tv::gemm::NumericArrayConverter<float, int32_t, 8, tv::gemm::FloatRoundStyle::round_to_nearest>
        accumulator_converter;
    tv::array<float, 8, 0> converted_accumulator = accumulator_converter(accumulator);
    tv::array<float, 8, 0> intermediate;
    tv::math::multiply_add<tv::array<float, 8, 0>> mul_add_accumulator;
    // alpha = output scale, beta = output_add scale
    intermediate = mul_add_accumulator(scale, converted_accumulator, bias); // intermediate = scale * converted_accumulator + bias_scaled
    intermediate = mul_add_accumulator(beta, converted_source, intermediate); // res = converted_source * beta + intermediate
    UnaryOp op;
    // activation
    intermediate = op(intermediate, act_type, act_alpha, act_beta);
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 720) && \
                            ((__CUDACC_VER_MAJOR__ > 10) ||                     \
                            ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2))))
    #else
      const float kClamp = float((1U << (sizeof(int8_t) * 8 - 1)) - 1);
      tv::math::minimum<tv::array<float, 8, 0>> min_op;
      tv::math::maximum<tv::array<float, 8, 0>> max_op;
      intermediate = max_op(intermediate, -kClamp - float(1));
      intermediate = min_op(intermediate, kClamp);
    #endif
    // Convert to destination numeric type
    tv::gemm::NumericArrayConverter<int8_t, float, 8, tv::gemm::FloatRoundStyle::round_to_nearest>
        destination_converter;
    return destination_converter(intermediate);
  }
  __forceinline__ __device__ tv::array<int8_t, 8, 0> operator()(tv::array<int32_t, 8, 0> const& accumulator, tv::array<float, 8, 0> const& bias, tv::array<float, 8, 0> const& scale)  const {
    
    tv::gemm::NumericArrayConverter<float, int32_t, 8, tv::gemm::FloatRoundStyle::round_to_nearest>
        accumulator_converter;
    tv::array<float, 8, 0> converted_accumulator = accumulator_converter(accumulator);
    tv::array<float, 8, 0> intermediate;
    tv::math::multiply_add<tv::array<float, 8, 0>> mul_add_accumulator;
    // alpha = output scale, beta = output_add scale
    intermediate = mul_add_accumulator(scale, converted_accumulator, bias); // intermediate = scale * converted_accumulator + bias_scaled
    UnaryOp op;
    // activation
    intermediate = op(intermediate, act_type, act_alpha, act_beta);
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 720) && \
                            ((__CUDACC_VER_MAJOR__ > 10) ||                     \
                            ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2))))
    #else
      const float kClamp = float((1U << (sizeof(int8_t) * 8 - 1)) - 1);
      tv::math::minimum<tv::array<float, 8, 0>> min_op;
      tv::math::maximum<tv::array<float, 8, 0>> max_op;
      intermediate = max_op(intermediate, -kClamp - float(1));
      intermediate = min_op(intermediate, kClamp);
    #endif
    // Convert to destination numeric type
    tv::gemm::NumericArrayConverter<int8_t, float, 8, tv::gemm::FloatRoundStyle::round_to_nearest>
        destination_converter;
    return destination_converter(intermediate);
  }
};
} // namespace out_op
} // namespace Ampere_s8s8s8s32f32tnt_m128n128k64m64n64k64A1T16832_300_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib