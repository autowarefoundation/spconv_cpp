#pragma once
#include <limits>
#include <spconvlib/cumm/common/TensorViewKernel.h>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/cumm/common/GemmBasic.h>
#include <spconvlib/spconv/csrc/utils/launch/LaunchUtils.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace maxpool {
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using LaunchUtils = spconvlib::spconv::csrc::utils::launch::LaunchUtils;
template <typename T, bool OneDim = false>
__global__ void forward_kernel(T* out_features, const T* in_features, const int* out_indices, const int* in_indices, int size, int num_features, int num_blocks_x, int num_blocks_y)   {
  
  int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
  int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;
  for (int i : tv::KernelLoopY<int>(size, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
      int in_idx = in_indices[i];
      int out_idx = out_indices[i];
      auto in_ptr = in_features + in_idx * num_features;
      auto out_ptr = out_features + out_idx * num_features;
      for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
          auto in = in_ptr[j];
          auto out = out_ptr[j];
          if (in > out){
              out_ptr[j] = in;
          }
      }
  }
}
template <typename T, bool OneDim = false>
__global__ void forward_implicit_gemm_kernel(T* out_features, const T* in_features, const int* indices, int num_features, int RS, int num_indices, T lowest, int num_blocks_x, int num_blocks_y)   {
  
  int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
  int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;
  for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
      auto out_ptr = out_features + i * num_features;
      for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
          auto indices_ptr = indices + i;
          int in_idx = indices_ptr[0];
          T in, in_temp;
          in = lowest;
          bool valid = in_idx != -1;
          in_temp = valid ? in_features[in_idx * num_features + j] : lowest;
          in = (in < in_temp && valid) ? in_temp: in;
          indices_ptr += num_indices;
          for (int k = 1; k < RS; ++k){
              in_idx = indices_ptr[0];
              valid = in_idx != -1;
              in_temp = valid ? in_features[in_idx * num_features + j] : lowest;
              in = (in < in_temp && valid) ? in_temp: in;
              indices_ptr += num_indices;
          }
          out_ptr[j] = in;
      }
  }
}
template <typename T, bool OneDim = false>
__global__ void backward_kernel(const T* out_features, const T* in_features, const T* dout_features, T* din_features, const int* out_indices, const int* in_indices, int size, int num_features, int num_blocks_x, int num_blocks_y)   {
  
  int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
  int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;
  for (int i : tv::KernelLoopY<int>(size, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
      int in_idx_offset = in_indices[i] * num_features;
      int out_idx_offset = out_indices[i] * num_features;
      auto in_ptr = in_features + in_idx_offset;
      auto out_ptr = out_features + out_idx_offset;
      auto din_ptr = din_features + in_idx_offset;
      auto dout_ptr = dout_features + out_idx_offset;
      for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
          auto in = in_ptr[j];
          auto out = out_ptr[j];
          if (in == out){
              din_ptr[j] = din_ptr[j] + dout_ptr[j];
          }
      }
  }
}
template <typename T, bool OneDim = false>
__global__ void backward_implicit_gemm_kernel(const T* out_features, const T* in_features, const T* dout_features, T* din_features, const int* indices_bwd, int num_features, int RS, int num_indices, int num_blocks_x, int num_blocks_y)   {
  
  int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
  int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;
  for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
      auto in_ptr = in_features + i * num_features;
      auto din_ptr = din_features + i * num_features;
      for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
          auto indices_ptr = indices_bwd + i;
          int out_idx = indices_ptr[0];
          T in = in_ptr[j];
          T sum_val = T(0);
          // if idx invalid, we only need to ensure in not equal to out.
          T out = out_idx != -1 ? out_features[out_idx * num_features + j] : T(0);
          T dout = out_idx != -1 ? dout_features[out_idx * num_features + j] : T(0);
          bool valid = in == out && out_idx != -1;
          sum_val = valid ? sum_val + dout : sum_val;
          indices_ptr += num_indices;
          for (int k = 1; k < RS; ++k){
              out_idx = indices_ptr[0];
              out = out_idx != -1 ? out_features[out_idx * num_features + j] : T(0);
              dout = out_idx != -1 ? dout_features[out_idx * num_features + j] : T(0);
              valid = in == out && out_idx != -1;
              sum_val = valid ? sum_val + dout : sum_val;
              indices_ptr += num_indices;
          }
          din_ptr[j] = sum_val;
      }
  }
}
template <typename T, bool OneDim = false>
__global__ void forward_avgpool_implicit_gemm_kernel(T* out_features, const T* in_features, const int* indices, int* count_out, int num_features, int RS, int num_indices, int num_blocks_x, int num_blocks_y)   {
  
  int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
  int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;
  for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
      auto out_ptr = out_features + i * num_features;
      auto indices_ptr = indices + i;
      int in_idx = 0;
      int count = 0;
      for (int k = 0; k < RS; ++k){
          in_idx = indices_ptr[0];
          count += int(in_idx != -1);
          indices_ptr += num_indices;
      }
      if (count_out != nullptr){
          count_out[i] = count;
      }
      for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
          indices_ptr = indices + i;
          int in_idx;
          T in, in_temp;
          in = T(0);
          for (int k = 0; k < RS; ++k){
              in_idx = indices_ptr[0];
              bool valid = in_idx != -1;
              in_temp = valid ? in_features[in_idx * num_features + j] : T(0);
              in += in_temp;
              indices_ptr += num_indices;
          }
          out_ptr[j] = count > 0 ? in / T(count) : T(0);
      }
  }
}
template <typename T, bool OneDim = false>
__global__ void backward_avgpool_implicit_gemm_kernel(const T* dout_features, T* din_features, const int* indices_bwd, const int* count_out, int num_features, int RS, int num_indices, int num_blocks_x, int num_blocks_y)   {
  
  int block_idx_x = OneDim ? blockIdx.x % num_blocks_x : blockIdx.x;
  int block_idx_y = OneDim ? blockIdx.x / num_blocks_x : blockIdx.y;
  for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, OneDim ? num_blocks_y : gridDim.y)) {
      auto din_ptr = din_features + i * num_features;
      for (int j : tv::KernelLoopX<int>(num_features, block_idx_x, OneDim ? num_blocks_x : gridDim.x)) {
          auto indices_ptr = indices_bwd + i;
          int out_idx = 0;
          T sum_val = T(0);
          for (int k = 0; k < RS; ++k){
              out_idx = indices_ptr[0];
              bool valid = out_idx != -1;
              T dout = valid ? dout_features[out_idx * num_features + j] : T(0);
              int count = valid ? count_out[out_idx] : T(0);
              sum_val += dout * T(count);
              indices_ptr += num_indices;
          }
          din_ptr[j] = sum_val;
      }
  }
}
/**
 * @param out_indices 
 * @param coords 
 * @param counts 
 * @param num_indices 
 * @param indices_stride 
 */
__global__ void global_pool_rearrange_kernel(int* out_indices, const int* coords, int* counts, int num_indices, int indices_stride);
struct IndiceMaxPool {
  static constexpr int kMaxGridYZDim = 65535;
  /**
   * @param out_indices 
   * @param coords 
   * @param counts 
   * @param stream 
   */
  static void global_pool_rearrange(tv::Tensor out_indices, tv::Tensor coords, tv::Tensor counts, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param in 
   * @param out_inds 
   * @param in_inds 
   * @param stream 
   */
  static void forward(tv::Tensor out, tv::Tensor in, tv::Tensor out_inds, tv::Tensor in_inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param in 
   * @param inds 
   * @param stream 
   */
  static void forward_implicit_gemm(tv::Tensor out, tv::Tensor in, tv::Tensor inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param in 
   * @param dout 
   * @param din 
   * @param out_inds 
   * @param in_inds 
   * @param stream 
   */
  static void backward(tv::Tensor out, tv::Tensor in, tv::Tensor dout, tv::Tensor din, tv::Tensor out_inds, tv::Tensor in_inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param in 
   * @param dout 
   * @param din 
   * @param inds 
   * @param stream 
   */
  static void backward_implicit_gemm(tv::Tensor out, tv::Tensor in, tv::Tensor dout, tv::Tensor din, tv::Tensor inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param in 
   * @param inds 
   * @param count_out 
   * @param stream 
   */
  static void forward_avgpool_implicit_gemm(tv::Tensor out, tv::Tensor in, tv::Tensor inds, tv::Tensor count_out, std::uintptr_t stream = 0);
  /**
   * @param dout 
   * @param din 
   * @param inds 
   * @param count_out 
   * @param stream 
   */
  static void backward_avgpool_implicit_gemm(tv::Tensor dout, tv::Tensor din, tv::Tensor inds, tv::Tensor count_out, std::uintptr_t stream = 0);
};
} // namespace maxpool
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib