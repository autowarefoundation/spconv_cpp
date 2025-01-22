#pragma once
#include <tensorview/gemm/math/all.h>
#include <tensorview/gemm/core/constants.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8f32s32f16tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8/out_op/unaryop/UnaryActivation.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8f32s32f16tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8 {
namespace out_op {
using UnaryOp = spconvlib::cumm::conv::main::Ampere_s8s8f32s32f16tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8::out_op::unaryop::UnaryActivation;
struct Int8Inference {
  tv::half_t alpha;
  tv::half_t beta;
  tv::half_t act_alpha;
  tv::half_t act_beta;
  tv::gemm::Activation act_type;
  __forceinline__ __device__  Int8Inference(tv::half_t alpha = tv::half_t(1), tv::half_t beta = tv::half_t(0), tv::half_t act_alpha = tv::half_t(0), tv::half_t act_beta = tv::half_t(0), tv::gemm::Activation type = tv::gemm::Activation::kNone) : alpha(alpha), beta(beta), act_alpha(act_alpha), act_beta(act_beta), act_type(type)  {
    
  }
  __forceinline__ __device__ bool is_source_needed()  const {
    return beta != tv::half_t(0);
  }
  __forceinline__ __device__ void set_k_partition(int k_part, int k_part_count)   {
    
    if (k_part) {
        // beta is for self reduce (part k from other wraps)
        beta = tv::half_t(1);
    }
  }
  __forceinline__ __device__ tv::array<float, 4, 0> operator()(tv::array<int32_t, 4, 0> const& accumulator, tv::array<float, 4, 0> const& source, tv::array<tv::half_t, 4, 0> const& bias, tv::array<tv::half_t, 4, 0> const& scale)  const {
    
    tv::gemm::NumericArrayConverter<tv::half_t, float, 4, tv::gemm::FloatRoundStyle::round_to_nearest>
        source_converter;
    tv::array<tv::half_t, 4, 0> converted_source = source_converter(source);
    tv::gemm::NumericArrayConverter<tv::half_t, int32_t, 4, tv::gemm::FloatRoundStyle::round_to_nearest>
        accumulator_converter;
    tv::array<tv::half_t, 4, 0> converted_accumulator = accumulator_converter(accumulator);
    tv::array<tv::half_t, 4, 0> intermediate;
    tv::math::multiply_add<tv::array<tv::half_t, 4, 0>> mul_add_accumulator;
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
    #endif
    // Convert to destination numeric type
    tv::gemm::NumericArrayConverter<float, tv::half_t, 4, tv::gemm::FloatRoundStyle::round_to_nearest>
        destination_converter;
    return destination_converter(intermediate);
  }
  __forceinline__ __device__ tv::array<float, 4, 0> operator()(tv::array<int32_t, 4, 0> const& accumulator, tv::array<tv::half_t, 4, 0> const& bias, tv::array<tv::half_t, 4, 0> const& scale)  const {
    
    tv::gemm::NumericArrayConverter<tv::half_t, int32_t, 4, tv::gemm::FloatRoundStyle::round_to_nearest>
        accumulator_converter;
    tv::array<tv::half_t, 4, 0> converted_accumulator = accumulator_converter(accumulator);
    tv::array<tv::half_t, 4, 0> intermediate;
    tv::math::multiply_add<tv::array<tv::half_t, 4, 0>> mul_add_accumulator;
    // alpha = output scale, beta = output_add scale
    intermediate = mul_add_accumulator(scale, converted_accumulator, bias); // intermediate = scale * converted_accumulator + bias_scaled
    UnaryOp op;
    // activation
    intermediate = op(intermediate, act_type, act_alpha, act_beta);
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 720) && \
                            ((__CUDACC_VER_MAJOR__ > 10) ||                     \
                            ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2))))
    #else
    #endif
    // Convert to destination numeric type
    tv::gemm::NumericArrayConverter<float, tv::half_t, 4, tv::gemm::FloatRoundStyle::round_to_nearest>
        destination_converter;
    return destination_converter(intermediate);
  }
};
} // namespace out_op
} // namespace Ampere_s8s8f32s32f16tnt_m64n64k32m32n32k32A1T16816_300_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib