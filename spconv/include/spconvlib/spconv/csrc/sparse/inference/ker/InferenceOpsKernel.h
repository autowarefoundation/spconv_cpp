#pragma once
#include <spconvlib/cumm/common/TensorViewKernel.h>
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
#include <spconvlib/cumm/common/GemmBasic.h>
#include <spconvlib/spconv/csrc/utils/launch/LaunchUtils.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace inference {
namespace ker {
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using LaunchUtils = spconvlib::spconv::csrc::utils::launch::LaunchUtils;
template <typename T, bool OneDim = false>
__global__ void bias_add_inplace_kernel(T* out_features, const T* bias, int size, int num_features, int num_blocks_x, int num_blocks_y)   {
  
  int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
  int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;
  for (int i : tv::KernelLoopY<int>(size, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
      auto out_ptr = out_features + i * num_features;
      for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
          out_ptr[j] = bias[j] + out_ptr[j];
      }
  }
}
template <typename T, bool OneDim = false>
__global__ void bias_add_act_inplace_kernel(T* out_features, const T* bias, tv::gemm::Activation act_type, T alpha, T beta, int size, int num_features, int num_blocks_x, int num_blocks_y)   {
  
  int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
  int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;
  namespace op = tv::arrayops;
  using nv_scalar_t = tv::equivalent_data_type_t<T>;
  using MathOp = op::MathScalarOp<nv_scalar_t>;
  for (int i : tv::KernelLoopY<int>(size, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
      auto out_ptr = out_features + i * num_features;
      for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
          T o = out_ptr[j] + bias[j];
          auto* o_nv = reinterpret_cast<nv_scalar_t*>(&o);
          switch (act_type){
              case tv::gemm::Activation::kNone:
                  break;
              case tv::gemm::Activation::kReLU:{
                  o = o >= T(0) ? o : T(0);
                  break;
              }
              case tv::gemm::Activation::kLeakyReLU:{
                  o = o >= T(0) ? o : o * alpha;
                  break;
              }
              case tv::gemm::Activation::kSigmoid:{
                  auto e = MathOp::exp(MathOp::neg(*o_nv));
                  o = T(1) / (T(1) + *reinterpret_cast<T*>( &e ));
                  break;
              }
              default: ;
          }
          out_ptr[j] = o;
      }
  }
}
template <typename T>
__global__ void activation_inplace_kernel(T* out_features, tv::gemm::Activation act_type, T alpha, T beta, int size)   {
  
  namespace op = tv::arrayops;
  using nv_scalar_t = tv::equivalent_data_type_t<T>;
  using MathOp = op::MathScalarOp<nv_scalar_t>;
  for (int i : tv::KernelLoopX<int>(size)) {
      T o = out_features[i];
      auto* o_nv = reinterpret_cast<nv_scalar_t*>(&o);
      switch (act_type){
          case tv::gemm::Activation::kNone:
              break;
          case tv::gemm::Activation::kReLU:{
              out_features[i] = o >= T(0) ? o : T(0);
              break;
          }
          case tv::gemm::Activation::kLeakyReLU:{
              out_features[i] = o >= T(0) ? o : o * alpha;
              break;
          }
          case tv::gemm::Activation::kSigmoid:{
              auto e = MathOp::exp(MathOp::neg(*o_nv));
              out_features[i] = T(1) / (T(1) + *reinterpret_cast<T*>( &e ));
              break;
          }
          default: ;
      }
  }
}
struct InferenceOpsKernel {
};
} // namespace ker
} // namespace inference
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib