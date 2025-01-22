#pragma once
#include <tensorview/cuda/launch.h>
#include <tensorview/cuda/kernel_utils.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace cudakers {
template <typename T>
__global__ void arange_kernel(T* data, int size)   {
  
  for (int i : tv::KernelLoopX<int>(size)) {
      data[i] = T(i);
  }
}
template <typename T>
__global__ void fill_kernel(T* data, T val, int size)   {
  
  for (int i : tv::KernelLoopX<int>(size)) {
      data[i] = T(val);
  }
}
template <typename T>
__global__ void maximum_value_kernel(T* data, T val, int size)   {
  
  for (int i : tv::KernelLoopX<int>(size)) {
      data[i] = max(data[i], val);
  }
}
struct CudaCommonKernel {
};
} // namespace cudakers
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib