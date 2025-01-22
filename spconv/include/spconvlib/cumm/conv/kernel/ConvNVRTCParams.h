#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace kernel {
struct ConvNVRTCParams {
  const void* ptr_A;
  const void* ptr_B;
  void* ptr_C;
  const void* ptr_D;
  float alpha;
  float beta;
  void* workspace;
  const uint32_t* mask_ptr;
  uint32_t* mask_out_ptr;
  uint32_t mask_filter;
  bool reverse_mask;
  int mask_width;
  int ndim;
  int N;
  int C;
  int K;
  tv::array<int, 3> input_dims;
  tv::array<int, 3> output_dims;
  tv::array<int, 3> ksize;
  tv::array<int, 3> padding;
  tv::array<int, 3> stride;
  tv::array<int, 3> dilation;
  int kernel_volume;
  tv::gemm::ConvMode mode;
  int split_k_slices;
  int groups;
};
} // namespace kernel
} // namespace conv
} // namespace cumm
} // namespace spconvlib