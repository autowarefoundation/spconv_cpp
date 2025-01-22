#pragma once
#include <tensorview/gemm/math/all.h>
#include <tensorview/gemm/core/constants.h>
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Simt_f32f32f32f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK {
namespace out_op {
namespace unaryop {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct UnaryActivation {
  __forceinline__ __device__ tv::array<float, 1, 0> operator()(const tv::array<float, 1, 0> & src, tv::gemm::Activation type, float alpha, float beta)   {
    
    namespace op = tv::arrayops;
    using scalar_nv_t = tv::equivalent_data_type_t<float>;
    auto& src_nv = reinterpret_cast<const tv::array<tv::equivalent_data_type_t<float>, 1, 0>&>(src);
    using MathOp = op::MathScalarOp<tv::equivalent_data_type_t<float>>;
    switch (type){
        case tv::gemm::Activation::kNone:
            return src;
        case tv::gemm::Activation::kReLU:{
            tv::math::maximum<tv::array<float, 1, 0>> max_op;
            return max_op(src, float(0));
        }
        case tv::gemm::Activation::kLeakyReLU:{
            tv::array<float, 1, 0> res;
            TV_PRAGMA_UNROLL
            for (int i = 0; i < 1; ++i){
                auto x = src[i];
                res[i] = x >= float(0) ? x : x * float(alpha);
            }
            return res;
        }
        case tv::gemm::Activation::kSigmoid:{
            tv::array<float, 1, 0> res;
            TV_PRAGMA_UNROLL
            for (int i = 0; i < 1; ++i){
                auto xx = MathOp::exp(MathOp::neg(src_nv[i]));
                res[i] = float(1) / (float(1) + *reinterpret_cast<float*>( &xx ));
            }
            return res;
        }
        default: return src;
    }
    return src;
  }
};
} // namespace unaryop
} // namespace out_op
} // namespace Simt_f32f32f32f32f32tnt_m64n256k8m32n64k8A1_200_C301LLL_SK
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib